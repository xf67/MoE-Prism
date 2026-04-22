import os
os.environ["HF_DATASETS_OFFLINE"] = "1" 

import torch
from torch import nn
from datasets import load_dataset

import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import random
import math

from transformers import PreTrainedModel
from tqdm import tqdm

from custom_models import RouterCompoundFast,NewCompoundMoE

shared_num_list = {'mixtral':0,'olmoe':0,'deepseek':2,'qwen':0} 
acti_num_list = {'mixtral':2,'olmoe':8,'deepseek':6,'qwen':8} 

class MLPWrapper(nn.Module):
    def __init__(self, module, offload='cpu'):
        super().__init__()
        self.module = module
        self.input_buffer = []
        self.acti_buffer = []
        self.offload = offload
    def forward(self, x):
        try:
            acti = self.module.act_fn(self.module.gate_proj(x)) * self.module.up_proj(x)
            output = self.module.down_proj(acti)
        except:
            acti = self.module.act_fn(self.module.w1(x)) * self.module.w3(x)
            output = self.module.w2(acti)

        self.input_buffer.append(x.float().to(self.offload))
        self.acti_buffer.append(acti.float().to(self.offload))
        torch.cuda.empty_cache()
        return output


class GateWrapper(nn.Module):
    def __init__(self, module, top_k):
        super().__init__()
        self.module_wrapped = module  
        self.gate_acti_buffer = []
        self.top_k = top_k

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = super().__getattr__('module_wrapped')
            try:
                return getattr(module, name)
            except AttributeError:
                raise AttributeError(
                    f"'{type(self).__name__}' and its wrapped module have no attribute '{name}'"
                )

    def forward(self, x):
        result = self.module_wrapped(x) 
        if isinstance(result, tuple):
            num_of_returns = len(result)
            if num_of_returns == 2: # our complex gate normal
                topk_weight, topk_idx = result
            elif num_of_returns == 3: # deepseek gate & our complex gate deepseek style
                topk_idx, topk_weight, aux_loss = result
        else: # linear
            router_logits = result
            output = router_logits
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, topk_idx = torch.topk(
                routing_weights, self.top_k, dim=-1
            )

        self.gate_acti_buffer.append(topk_idx.to("cpu"))
        # print(topk_idx)
        torch.cuda.empty_cache()
        return result

    


def get_wikitext2(nsamples, seed, seqlen, tokenizer_id, bsz = 1, max_batch_num=128):
    # 加载数据集
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    # 加载 tokenizer
    if isinstance(tokenizer_id, str):
        from transformers import AutoTokenizer 
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    else:
        tokenizer = tokenizer_id

    # 逐条文本进行 tokenize，拼接成一个长的 input_ids 序列
    all_ids = []
    for text in traindata['text']:
        if text.strip() == '':
            continue  # 跳过空行
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(tokens)

    total_length = len(all_ids)
    print("Total token length:", total_length)

    # 划分成 batch，每个样本长 seqlen
    trainloader = []
    for i in range(0, total_length - seqlen * bsz + 1, seqlen * bsz):
        if i // (seqlen * bsz) >= max_batch_num:
            break

        batch_inputs = []
        for j in range(bsz):
            start_idx = i + j * seqlen
            end_idx = start_idx + seqlen
            input_ids = torch.tensor(all_ids[start_idx:end_idx]).unsqueeze(0)  # [1, seqlen]
            batch_inputs.append(input_ids)

        batch = torch.cat(batch_inputs, dim=0)  # [bsz, seqlen]
        trainloader.append(batch)

    return trainloader, None  # 第二项是 testloader，占位

def get_wikitext2_old(nsamples, seed, seqlen, tokenizer_id, bsz = 1, max_batch_num=128): #原来那版是什么玩意，冗余得要死？
    # print('get_wikitext2:', seqlen, tokenizer_id, bsz, max_batch_num)
    
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    # testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    if isinstance(tokenizer_id,str): 
        from transformers import AutoTokenizer 
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True) #原来是false，但是olmoe用的那版没有不用fast的版本
    else:
        tokenizer = tokenizer_id #也可以直接传tokenizer进来用

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    # testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    total_length = trainenc.input_ids.shape[1]
    print("total length:" ,  total_length)
    trainloader = []
    for i in range(0, total_length - seqlen + 1, seqlen * bsz):
        if i/seqlen >= max_batch_num:
            break
        batch_inputs = []

        for j in range(bsz):
            start_idx = i + j * seqlen
            end_idx = start_idx + seqlen

            # 防止结尾不足一个序列长度
            if end_idx > total_length:
                break

            inp = trainenc.input_ids[:, start_idx:end_idx]
            batch_inputs.append(inp)

        if len(batch_inputs) == bsz:  # 只有当batch内样本数完整时, 才加入loader
            batch_i = torch.cat(batch_inputs, dim=0)
            trainloader.append(batch_i)

    return trainloader,None #这个None是代替了test_loader，反正我们不用




@torch.no_grad()
def balance_neuros(cross_activate,partitions,num_partitions): 
    #python对可变对象传的是引用，所以在函数内修改即可

    target_count = cross_activate.shape[0] // num_partitions

    oversize_partitions = []
    undersize_partitions = []

    for pid in range(num_partitions):
        actual_size = len(partitions[pid])
        if actual_size > target_count:
            oversize_partitions.append((pid, actual_size - target_count))
        elif actual_size < target_count:
            undersize_partitions.append((pid, target_count - actual_size))

    for oversize_pid, excess_count in oversize_partitions:
        for _ in range(excess_count):
            best_node = None
            best_undersize_pid = None
            min_similarity_to_current = np.inf  # 初始化为正无穷，寻找最小的相似度

            # 遍历当前过大的组内所有节点，选择一个节点移动
            for node in partitions[oversize_pid]:
                # 计算此节点与当前组内所有节点的相似度总和
                sim_score = cross_activate[node, partitions[oversize_pid]].sum()
                if sim_score < min_similarity_to_current:
                    min_similarity_to_current = sim_score
                    best_node = node

            # 为选出的节点选择一个欠节点的分组
            max_similarity_to_target = -np.inf
            for undersize_pid, lack_count in undersize_partitions:
                # 计算此节点与目标组内所有节点的相似度总和
                sim_score = cross_activate[best_node, partitions[undersize_pid]].sum()
                if sim_score > max_similarity_to_target:
                    max_similarity_to_target = sim_score
                    best_undersize_pid = undersize_pid

            # 执行节点移动
            partitions[oversize_pid].remove(best_node)
            partitions[best_undersize_pid].append(best_node)

            # 更新需要补充节点数量
            for idx, (upid, lack_count) in enumerate(undersize_partitions):
                if upid == best_undersize_pid:
                    if lack_count == 1:
                        undersize_partitions.pop(idx)
                    else:
                        undersize_partitions[idx] = (upid, lack_count - 1)
                    break

@torch.no_grad()
def wrap_metis(cross_activate,threshold,num_partitions):
    import pymetis
    cross_activate_slim = torch.triu(cross_activate, diagonal=1)

    i_idx, j_idx = np.where(cross_activate_slim > threshold)
    #建图，还是单线程，GPT写的多线程不知道哪里有bug
    xadj = [0]  # xadj 的初始值为 0
    adjncy = []  # 存储所有邻居节点
    eweights = []  # 存储所有边的权重
    for i in range(cross_activate_slim.shape[0]):
        # 找到节点 i 的所有邻居
        neighbors = j_idx[i_idx == i]
        # 添加邻居节点到 adjncy
        adjncy.extend(neighbors)
        # 添加边的权重到 eweights
        eweights.extend(cross_activate_slim[i, neighbors])
        # 更新 xadj
        xadj.append(len(adjncy))

    #metis
    edge_cuts, membership = pymetis.part_graph(num_partitions, xadj=xadj,adjncy=adjncy,eweights=eweights)
    partitions = {i: [] for i in range(num_partitions)}
    for node, part_id in enumerate(membership):
        partitions[part_id].append(node)

    del(xadj,adjncy,eweights,i_idx,j_idx)
    #调节节点数平衡
    balance_neuros(cross_activate,partitions,num_partitions)
    return partitions

@torch.no_grad()
def wrap_kmeans(binary_mask,Wg_remain,Wu_remain,W_centroid,neuron_size,num_partitions,kmeansI=10,alpha=1):
    import lap
    binary_mask=binary_mask.cuda()
    Wg_remain=Wg_remain.float().cuda()
    Wu_remain=Wu_remain.float().cuda()
    activate_rates=torch.sum(binary_mask,dim=0) #1024
    W_=torch.concat([Wg_remain,Wu_remain],dim=-1)
    dis=torch.cdist(W_,W_centroid,p=1) # 1024,num_parti
    top_indices = torch.argmax(dis, dim=0)
    del W_
    # _, top_indices = torch.topk(activate_rates, num_partitions)
    indices_list = list(range(neuron_size))
    selected_index = top_indices
    cluster_size = neuron_size//num_partitions
    centroidsH = binary_mask[:, selected_index].clone() # bstok*sele
    centroidsG = Wg_remain[selected_index].clone() # sele*2048
    centroidsU = Wu_remain[selected_index].clone() # sele*2048
    distance_weight = activate_rates[selected_index]/torch.mean(activate_rates[selected_index]) #sele
    max_iters = kmeansI
    prev_assignments = None
    for iteration in range(max_iters):
            
            distances_H = torch.cdist(binary_mask[:, indices_list].T, centroidsH.T, p=1) 
            distances_G = torch.cdist(Wg_remain[indices_list], centroidsG, p=2) 
            distances_U = torch.cdist(Wu_remain[indices_list], centroidsU, p=2) 
            distances= ((distances_G+distances_U)*alpha + distances_H*distance_weight)/(distance_weight+1)
            distances_np = distances.cpu().numpy()
            repeated_distances = np.zeros((neuron_size, neuron_size))
            for i in range(num_partitions):
                repeated_distances[:, i*cluster_size:(i+1)*cluster_size] = distances_np[:, i:i+1]
                
            row_ind, col_ind = lap.lapjv(repeated_distances)[0:2]  #LAPJV算法
            assignments = torch.tensor(col_ind // cluster_size)

            if prev_assignments is not None and torch.all(assignments == prev_assignments): #提前退出
                break
            prev_assignments = assignments.clone()
            
            #更新质心
            for i in range(num_partitions):
                cluster_points = binary_mask[:, indices_list][:, assignments == i]
                cluster_pointsG = Wg_remain[indices_list][assignments == i]
                cluster_pointsU = Wu_remain[indices_list][assignments == i]
                if cluster_points.size(1) > 0:
                    centroidsH[:, i] = cluster_points.mean(dim=1)
                    centroidsG[i] = cluster_pointsG.mean(dim=0)
                    centroidsU[i] = cluster_pointsU.mean(dim=0)
    membership = prev_assignments.detach().cpu().numpy()
    partitions = {i: [] for i in range(num_partitions)}
    for node, part_id in enumerate(membership):
        partitions[part_id].append(node)
    return partitions


@torch.no_grad()
def wrap_kmeans_simple(
    markers: torch.Tensor,
    num_partitions: int,
    neurons_per_expert: int,
    max_iters: int = 10
) -> list[list[int]]:
    """
    使用一种平衡的K-Means算法将神经元划分为大小相等的专家组。
    此版本经过优化，可最大程度地利用CUDA，但受限于基于CPU的lapjv求解器。

    该算法通过线性分配问题（LAP）求解器确保每个簇的大小完全相等，
    这与标准K-Means不同，后者不保证簇大小。

    Args:
        markers (torch.Tensor): 形状为 (B, C) 的激活矩阵，其中B是样本数，C是神经元总数。
                                应已放置在目标设备（如'cuda'）上。
        num_experts (int): 目标专家组（簇）的数量，即 K。
        neurons_per_expert (int): 每个专家组应包含的神经元数量。
        max_iters (int): K-Means迭代的最大次数。

    Returns:
        list[list[int]]: 包含每个专家组神经元原始索引的列表。

    NOTE:
        - 性能瓶颈: `lap.lapjv` 是一个CPU密集型操作，具有 O(N^3) 的复杂度，
          其中N是要聚类的神经元数量。这部分需要将距离矩阵从GPU移至CPU。
        - 内存警告: 为 `lapjv` 构建的成本矩阵大小为 (N, N)，当N很大时会消耗大量内存。
    """
    import lap
    device = markers.device
    
    # --- 1. 预处理和设置 (在GPU上) ---
    remaining_indices = torch.arange(markers.shape[1], device=device)

    # 仅提取我们需要聚类的神经元的激活数据
    points_to_cluster = markers
    num_points = len(remaining_indices)

    if num_points != num_partitions * neurons_per_expert:
        raise ValueError("要聚类的神经元总数必须等于 num_experts * neurons_per_expert")

    # --- 2. 初始化质心 (在GPU上) ---
    # 策略：选择激活率最高的K个神经元作为初始质心
    activation_rates = points_to_cluster.mean(dim=0)
    _, top_indices_in_subset = torch.topk(activation_rates, num_partitions)
    
    # 直接从points_to_cluster中高效地收集初始质心
    centroids = points_to_cluster[:, top_indices_in_subset].clone()

    # --- 3. K-Means 迭代循环 ---
    prev_assignments = None
    for iteration in range(max_iters):
        
        # --- 3a. 分配步骤 ---
        # 计算所有点到所有质心的L1距离 (在GPU上完成)
        # points_to_cluster.T: (num_points, B)
        # centroids.T: (k, B)
        # distances: (num_points, k)
        distances = torch.cdist(points_to_cluster.T, centroids.T, p=1)

        # --- !! 性能瓶颈开始 !! ---
        # lapjv 需要一个 (N, N) 的成本矩阵，并且在CPU上运行
        # 我们将距离矩阵复制到CPU
        distances_cpu = distances.to('cpu', non_blocking=True).numpy()
        
        # 构建lapjv所需的成本矩阵
        # cost_matrix[i, j] 是将点i分配给簇k的成本，其中 j = k*size + m
        cost_matrix = np.repeat(distances_cpu, neurons_per_expert, axis=1)
        
        # 使用LAPJV求解器找到最优的平衡分配
        # 这是一个 O(N^3) 的操作，是整个流程中最慢的部分
        _, col_ind, _ = lap.lapjv(cost_matrix, extend_cost=True)
        
        # 将分配结果转换回簇索引，并移回GPU
        assignments = torch.from_numpy(col_ind // neurons_per_expert).to(device)
        # --- !! 性能瓶颈结束 !! ---

        # 检查是否收敛
        if prev_assignments is not None and torch.all(assignments == prev_assignments):
            # print(f"\nConvergence reached at iteration {iteration+1}.")
            break
        prev_assignments = assignments.clone()
        
        # --- 3b. 更新质心步骤 (向量化，在GPU上完成) ---
        # 使用 one-hot 编码和矩阵乘法高效更新质心，避免 for 循环
        
        # (num_points, k)
        assignments_one_hot = torch.nn.functional.one_hot(assignments.to(torch.long), num_classes=num_partitions)
        
        # 计算每个簇中点的数量 (应该总是 neurons_per_expert)
        points_per_cluster = assignments_one_hot.sum(dim=0) # shape: (k,)
        
        # 计算每个簇中所有点的向量和
        # (B, num_points) @ (num_points, k) -> (B, k)
        sum_of_points = points_to_cluster @ assignments_one_hot.to(points_to_cluster.dtype)
        
        # 计算新的质心（均值）
        # 使用 clamp 避免除以零（尽管在此平衡场景中不太可能发生）
        centroids = sum_of_points / points_per_cluster.clamp(min=1)

    # --- 4. 生成最终的专家组 (在GPU上) ---
    expert_groups = []
    final_assignments_cpu = assignments.to('cpu')
    for i in range(num_partitions):
        # 找到属于当前簇的点的索引 (在子集内的索引)
        cluster_mask = final_assignments_cpu == i
        indices_in_subset = torch.where(cluster_mask)[0]
        
        # 将子集索引映射回原始的神经元索引
        original_indices = remaining_indices[indices_in_subset].tolist()
        expert_groups.append(original_indices)
        
    return expert_groups


def calculate_total_cost_torch_l1(K, group_norms_l1):
    """
    使用 PyTorch 高效地计算 L1 范数下的总成本。
    
    Args:
        K (int): 每行选取的最小范数组数。
        group_norms_l1 (torch.Tensor): 一个 B x N 的张量，直接包含每个组的L1范数。

    Returns:
        float: 总成本（标量）。
    """
    # L1范数下，group_norms_l1 已经包含了我们需要的范数值，无需开方。
    k_smallest_norms, _ = torch.topk(group_norms_l1, k=K, dim=1, largest=False)
    return torch.sum(k_smallest_norms).item()


def greedy_initializer_torch_l1(matrix_abs, N, group_size, device):
    """
    使用 PyTorch 和 L1 范数实现启发式贪心算法。
    
    Args:
        matrix_abs (torch.Tensor): 预计算的矩阵绝对值，在指定设备上。
        N (int): 分组数。
        group_size (int): 每个组的大小。
        device: PyTorch device ('cuda' or 'cpu').

    Returns:
        list[list[int]]: 初始分区。
        torch.Tensor: B x N 的初始L1范数张量。
    """
    B = matrix_abs.shape[0]
    
    # 1. 计算每列的L1范数（即绝对值之和）
    col_l1_sums = torch.sum(matrix_abs, dim=0)
    
    # 2. 按L1范数从大到小排序
    sorted_cols = torch.argsort(col_l1_sums, descending=True)
    
    # 3. 初始化
    partition = [[] for _ in range(N)]
    # 该张量现在直接存储L1范数
    group_norms_l1 = torch.zeros((B, N), dtype=torch.float64, device=device)
    group_current_sizes = torch.zeros(N, dtype=torch.int, device='cpu')
    
    # 4. 逐个分配列
    for col_idx in sorted_cols.tolist():
        available_groups_mask = group_current_sizes < group_size
        valid_indices = torch.where(available_groups_mask)[0]
        
        # 计算有效组的总L1范数（所有B行范数之和）
        group_total_l1_norms = torch.sum(group_norms_l1[:, valid_indices], dim=0)
        
        min_norm_idx_local = torch.argmin(group_total_l1_norms)
        target_group_idx = valid_indices[min_norm_idx_local].item()
        
        # 分配并更新
        partition[target_group_idx].append(col_idx)
        # 直接加上该列的绝对值向量
        group_norms_l1[:, target_group_idx] += matrix_abs[:, col_idx]
        group_current_sizes[target_group_idx] += 1
        
    return partition, group_norms_l1

@torch.no_grad()
def wrap_sa(
    matrix: torch.Tensor, 
    N: int, 
    K: int,
    max_iter: int = 500000,
    initial_temp: float = 100.0,  # L1范数值域可能不同，温度可适当调整
    cooling_rate: float = 0.99995
):
    """
    使用 PyTorch 和 L1 范数解决列分组优化问题。

    Args:
        matrix (torch.Tensor): 输入的 B x C 张量。
        N (int): 分组数。
        K (int): 每行选取的最小范数组数。
        max_iter (int): 模拟退火最大迭代次数。
        initial_temp (float): 初始温度。
        cooling_rate (float): 冷却率。

    Returns:
        tuple[list[list[int]], float]:
            - P (list[list[int]]): 最优分区。
            - loss (float): 最小总L1范数和。
    """
    device = matrix.device
    B, C = matrix.shape
    
    if C % N != 0:
        raise ValueError(f"列数 C ({C}) 必须能被分组数 N ({N}) 整除。")
    
    # print(f"Running on device: {device} with L1 Norm")
    # print("Pre-computation and Initialization...")
    
    # 预先计算整个矩阵的绝对值
    matrix_abs = torch.abs(matrix).to(dtype=torch.float64)

    # 1. 贪心初始化
    P_current, norms_l1_current = greedy_initializer_torch_l1(matrix_abs, N, C // N, device)
    cost_current = calculate_total_cost_torch_l1(K, norms_l1_current)
    
    best_P = [list(g) for g in P_current]
    best_cost = cost_current
    
    temp = initial_temp
    
    # print(f"Starting Greedy solution cost (L1): {best_cost:.4f}")
    # print("Running Simulated Annealing...")
    
    # 2. 模拟退火
    for i in range(max_iter):
        try:
            g1_idx, g2_idx = random.sample(range(N), 2)
            if not P_current[g1_idx] or not P_current[g2_idx]: continue
            col1 = random.choice(P_current[g1_idx])
            col2 = random.choice(P_current[g2_idx])
        except (ValueError, IndexError):
            continue

        # 增量更新L1范数
        col1_abs_vec = matrix_abs[:, col1]
        col2_abs_vec = matrix_abs[:, col2]
        
        norms_l1_new = norms_l1_current.clone()
        norms_l1_new[:, g1_idx] += col2_abs_vec - col1_abs_vec
        norms_l1_new[:, g2_idx] += col1_abs_vec - col2_abs_vec
        
        cost_new = calculate_total_cost_torch_l1(K, norms_l1_new)
        
        # 接受准则
        delta_cost = cost_new - cost_current
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / temp):
            P_current[g1_idx].remove(col1)
            P_current[g1_idx].append(col2)
            P_current[g2_idx].remove(col2)
            P_current[g2_idx].append(col1)
            
            norms_l1_current = norms_l1_new
            cost_current = cost_new
            
            if cost_current < best_cost:
                best_cost = cost_current
                best_P = [list(g) for g in P_current]
                
        temp *= cooling_rate

    # print(f"\nOptimization finished.")
    # print(f"Final best cost (L1): {best_cost:.4f}")
    
    return best_P

@torch.no_grad()
def wrap_rand(tol_len,part):

    numbers = list(range(tol_len))

    random.shuffle(numbers)

    group_size = tol_len // part

    result = [numbers[i : i + group_size] for i in range(0, tol_len, group_size)]

    return result



def get_compound_with_biggerGate(model:PreTrainedModel, cfg): #inner_num表示内层拆成几个sub-expert
    model.cpu()
    if model.config.architectures==["MixtralForCausalLM"]:
        name='mixtral'
    elif model.config.architectures==["OlmoeForCausalLM"]:
        name='olmoe'
    elif model.config.architectures==["DeepseekV2ForCausalLM"]:
        name='deepseek'
    elif model.config.architectures==["Qwen3MoeForCausalLM"]:
        name='qwen'    
    assert name in ['mixtral', 'olmoe', 'deepseek', 'qwen']
    represent_moe_layer=model.model.layers[1].block_sparse_moe if name=='mixtral' else model.model.layers[1].mlp
    represent_sub_mode_layer=represent_moe_layer.experts[0] #MoE(gate is RouterBigger, experts[:] is LLamaMLP)
    for i in tqdm(range(len(model.model.layers)), desc = 'Composite MoE layers...'): #这层是一层decoder layer
        layer = model.model.layers[i]
        moe=layer.block_sparse_moe if name=='mixtral' else layer.mlp
        try:
            out_gate=moe.gate
            if name=='deepseek':
                out_gate.early_output=True
        except:
            print(f"Layer {layer} is not a MoE layer, skip.")
            continue
        in_gates_list=nn.ModuleList()
        experts_list=nn.ModuleList()
        for out_expert in moe.experts: #这是原网络的一个expert
            in_gates_list.append(out_expert.gate) #一共有expert_num个
            experts_list.extend(out_expert.experts) #一共有expert_num*sub_expert_num个
        # 下面这个初始化还得改
        
        new_gate=RouterCompoundFast(cfg)
        new_gate.init_weight(out_gate,in_gates_list)
        if name=='deepseek':
            new_gate.deepseek_style = True 
            layer.mlp.experts = experts_list
            layer.mlp.gate = new_gate
        else:
            raise(NotImplementedError)

