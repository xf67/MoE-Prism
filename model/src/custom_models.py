import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

class RouterBigger(nn.Module):
    def __init__(self, hidden_size, n_experts, n_activated, bigger_size=1, bias_speed = 0.001):
        super().__init__()
        self.dim = hidden_size
        self.topk = n_activated

        self.act_fn = F.silu
        self.gate = nn.Linear(hidden_size, n_experts*bigger_size, bias=False)
        self.up = nn.Linear(hidden_size, n_experts*bigger_size, bias=False)

        self.extra_scale = nn.Parameter(torch.zeros(n_experts, device='cuda', dtype=torch.bfloat16))
        self.extra_bias = torch.zeros(n_experts, device='cuda', dtype=torch.float32)
        self.bias_update_speed = bias_speed
        self.n_experts=n_experts
        self.bigger_size=bigger_size
    
    def update_bias(self, counts):
        mean_load = counts.mean()
        # Decrease bias for overloaded experts, increase for underloaded
        overloaded = counts > mean_load
        underloaded = counts < mean_load

        self.extra_bias.data[overloaded] -= self.bias_update_speed
        self.extra_bias.data[underloaded] += self.bias_update_speed

    def save_logits(self, logits):
        # 将 logits 保存到文件
        with open('moe_gate_logits_ori_c.txt', 'a') as f:
            logits_np = logits.float().cpu().detach().numpy()
            for logit in logits_np:
                f.write(','.join(map(str, logit)) + '\n')

    def get_score(self, x: torch.Tensor) -> torch.Tensor:
        scores = (self.up(x) * self.act_fn(self.gate(x))).abs() 
        scores = scores.view(-1,self.n_experts,self.bigger_size)
        scores = scores.mean(dim=2)
        
        return scores

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        scores = self.get_score(x)
        scores = scores.softmax(dim=-1, dtype=torch.float32)
        original_scores = scores
        scores = scores + self.extra_bias[None, :]

        indices = torch.topk(scores, self.topk, dim=-1)[1]

        original_scores = 1 + original_scores*self.extra_scale
        weights = original_scores.gather(1, indices)

        # self.save_logits(weights)

        return weights.type_as(x), indices
    

class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu"
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu if hidden_act == "silu" else getattr(F, hidden_act)

    def forward(self, x):
        intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        output = self.down_proj(intermediate)
        return output



class MoE(nn.Module):
    def __init__(self, hidden_size, moe_inter_dim, n_experts, n_shared, n_activated, bigger):
        super().__init__()
        self.dim = hidden_size
        n_routed_experts = n_experts - n_shared
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated
        self.experts_start_idx = 0
        self.experts_end_idx = n_routed_experts
        self.gate = RouterBigger(hidden_size, n_routed_experts, self.n_activated_experts, bigger)
        self.n_shared_experts = n_shared
        self.experts = nn.ModuleList([LlamaMLP(self.dim, moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = LlamaMLP(self.dim, self.n_shared_experts * moe_inter_dim) if n_shared > 0 else None
        # self.shared_experts = None #先丢掉shared

        self.cus_training = False
        self.enable_scale = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        # print(shape) # 1x94x4096
        x = x.view(-1, self.dim)
        if self.n_activated_experts!=0:
            weights, indices = self.gate(x) #门控的计算,indices是激活的专家的索引
            # print(weights,indices) # weights 都是1，indices是激活的专家索引2~15
            # print(weights.shape,indices.shape) # 都是94x2(2表示激活两个专家)
            y = torch.zeros_like(x)
            counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts) #统计了每个专家被激活的次数，即对indicies进行统计
            if self.cus_training:
                self.gate.update_bias(counts.to(dtype = torch.bfloat16))
            counts = counts.tolist()
            for i in range(self.experts_start_idx, self.experts_end_idx): #for循环遍历专家
                if counts[i] == 0: #如果专家没有被激活，就跳过
                    continue
                expert = self.experts[i]
                idx, top = torch.where(indices == i) #找到激活的函数所在的坐标
                xx=x[idx] #筛选出来要计算的token
                if self.enable_scale:
                    y[idx] += expert(xx) * weights[idx, top, None]
                else:
                    y[idx] += expert(xx) 
            if self.shared_experts:
                z = self.shared_experts(x)
                return (y + z).view(shape)
            else:
                return y.view(shape)
        else:
            z = self.shared_experts(x)
            return z.view(shape)


class RouterCompound(nn.Module):
    def __init__(self, out_gate, in_gates_list, inner_num, acti_num, acti_pattern,scale_factor=1.0):
        super().__init__()
        self.norm_topk_prob:bool = True
        self.out_gate:nn.Linear = out_gate
        self.in_gates:nn.ModuleList = in_gates_list
        self.inner_num:int = inner_num
        self.acti_num:int = acti_num
        self.acti_pattern:list = acti_pattern
        self.routed_scaling_factor: float = scale_factor
        assert len(self.acti_pattern) == acti_num
        # self.no_cross_activate=False

    def forward(self, x: torch.Tensor) -> torch.Tensor: #用排序+mask的方式实现动态topk，感谢GPT
        bs, dim = x.shape
        device = x.device

        # Step 1: Outer expert selection
        # print(x)
        out = self.out_gate(x)
        # print(out)
        
        out_scores = F.softmax(out, dim=-1, dtype=torch.float32)  # (B, E)
        routing_weights, selected_experts = torch.topk(out_scores, self.acti_num, dim=-1)  # (B, K)

        if self.norm_topk_prob: 
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        else:
            routing_weights *= self.routed_scaling_factor

        # Step 2: Prepare (B*K, D) inputs and (B*K,) selected expert ids
        expanded_x = x.unsqueeze(1).expand(-1, self.acti_num, -1)     # (B, K, D)，expand实际上是复制了unsqueeze得到的中间那个维度K次
        flat_x = expanded_x.reshape(-1, dim)                          # (B*K, D)
        flat_expert_ids = selected_experts.reshape(-1)               # (B*K,)
        flat_weights = routing_weights.reshape(-1)                   # (B*K,)

        # Step 3: Distribute inputs to each expert
        expert_masks = [(flat_expert_ids == i) for i in range(len(self.in_gates))] 
            #ingates有num_expert个，用来route sub-expert。上面这个mask是 E*(B*K), bool

        all_inner_scores = torch.zeros(flat_x.shape[0], self.inner_num, device=device, dtype=x.dtype) # B*K * sub_expert_num

        for eid, mask in enumerate(expert_masks):
            if mask.any():
                inputs = flat_x[mask]  # (N_i, D)
                scores = self.in_gates[eid].get_score(inputs)  # (N_i, inner_num)
                all_inner_scores[mask] = scores

        # Step 4: Inner top-k routing (B*K, inner_num) → selected indices
        # Build pattern mask: each K has corresponding pattern length
        inner_topks = torch.tensor(self.acti_pattern, device=device)  # (K,)
        inner_topks = inner_topks.unsqueeze(0).expand(bs, -1).reshape(-1)  # (B*K,)

        max_topk = max(self.acti_pattern)
        topk_values, topk_indices = torch.topk(all_inner_scores, k=max_topk, dim=-1)  # (B*K, max_k)

        # Mask out unused values
        arange = torch.arange(max_topk, device=device).unsqueeze(0)  # (1, max_k)
        mask = arange < inner_topks.unsqueeze(1)                     # (B*K, max_k)

        # Select valid indices and weights
        flat_expert_ids_expanded = flat_expert_ids.unsqueeze(1).expand(-1, max_topk) # (B*K, max_k)
        selected_inner_ids = flat_expert_ids_expanded * self.inner_num + topk_indices  # (B*K, max_k)
        selected_inner_ids = selected_inner_ids.clone() 
        selected_inner_ids[~mask] = 0
        selected_inner_ids = selected_inner_ids[mask].view(bs, -1)

        selected_weights = flat_weights.unsqueeze(1).expand(-1, max_topk)
        selected_weights = selected_weights.clone()
        selected_weights[~mask] = 0
        selected_weights = selected_weights[mask].view(bs, -1)

        # print(selected_experts)
        # exit()
        return selected_weights, selected_inner_ids
    


    

# --- Triton Kernel ---
try:
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_D': 64}, num_warps=2),
            triton.Config({'BLOCK_SIZE_D': 128}, num_warps=4),
            triton.Config({'BLOCK_SIZE_D': 256}, num_warps=4),
            triton.Config({'BLOCK_SIZE_D': 512}, num_warps=8),
        ],
        key=['D', 'B_SIZE'],
    )
    @triton.jit
    def router_forward_kernel(
        # --- Pointers to Tensors ---
        X_ptr,
        Stacked_Gate_Weights_ptr,
        Stacked_Up_Weights_ptr,
        Expert_IDs_ptr,
        Out_Scores_ptr,
        # --- Tensor dimensions ---
        D, E_out, E_in, B_SIZE,
        # --- Strides ---
        stride_x_bk, stride_x_d,
        stride_gw_eout, stride_gw_ein_bsize, stride_gw_d,
        stride_uw_eout, stride_uw_ein_bsize, stride_uw_d,
        stride_eid_bk,
        stride_os_bk, stride_os_ein,
        # --- Meta-parameters ---
        BLOCK_SIZE_D: tl.constexpr, # Block size for D dimension
    ):
        """
        Triton Kernel to fuse the inner expert scoring logic.
        Grid: (B*K, E_in)
        """
        # --- Get program IDs to identify the current instance ---
        pid_token = tl.program_id(axis=0)  # Current token index (0 to B*K-1)
        pid_inner_expert = tl.program_id(axis=1)  # Current inner expert index (0 to E_in-1)

        # --- Load the outer expert ID for the current token ---
        # Expert_IDs_ptr is of shape (B*K), so we just need pid_token
        expert_id_ptr = Expert_IDs_ptr + pid_token * stride_eid_bk
        outer_expert_id = tl.load(expert_id_ptr)

        # --- Initialize accumulator for the mean score ---
        total_score_acc = 0.0

        # --- Loop over the 'bigger_size' dimension ---
        # Each inner expert has B_SIZE sub-vectors
        for b_idx in range(B_SIZE):
            # --- Calculate dot product for Gate and Up weights ---
            gate_acc = tl.zeros((), dtype=tl.float32)
            up_acc = tl.zeros((), dtype=tl.float32)

            # Loop over the hidden dimension D in blocks
            for d_start in range(0, tl.cdiv(D, BLOCK_SIZE_D)):
                d_offsets = d_start * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
                d_mask = d_offsets < D

                # 1. Load a block of the input vector x
                x_ptr = X_ptr + pid_token * stride_x_bk + d_offsets * stride_x_d
                x_block = tl.load(x_ptr, mask=d_mask, other=0.0)

                # 2. Calculate offset for the current weight row
                # Weight row index = inner_expert_idx * B_SIZE + b_idx
                weight_row_offset = (pid_inner_expert * B_SIZE + b_idx) * stride_gw_ein_bsize

                # 3. Load a block of the gate weight vector
                gate_weight_ptr = (Stacked_Gate_Weights_ptr +
                                   outer_expert_id * stride_gw_eout +
                                   weight_row_offset +
                                   d_offsets * stride_gw_d)
                gate_weight_block = tl.load(gate_weight_ptr, mask=d_mask, other=0.0)

                # 4. Load a block of the up weight vector
                up_weight_ptr = (Stacked_Up_Weights_ptr +
                                 outer_expert_id * stride_uw_eout +
                                 weight_row_offset + # same offset logic
                                 d_offsets * stride_uw_d)
                up_weight_block = tl.load(up_weight_ptr, mask=d_mask, other=0.0)

                # 5. Accumulate dot product
                gate_acc += tl.sum(x_block * gate_weight_block)
                up_acc += tl.sum(x_block * up_weight_block)

            # --- Apply activation and calculate score for this sub-vector ---
            # silu(x) = x * sigmoid(x)
            silu_gate = gate_acc * tl.sigmoid(gate_acc)
            sub_score = tl.abs(silu_gate * up_acc)

            # Accumulate the score
            total_score_acc += sub_score

        # --- Calculate the mean score ---
        mean_score = total_score_acc / B_SIZE

        # --- Write the final score to the output tensor ---
        out_ptr = Out_Scores_ptr + pid_token * stride_os_bk + pid_inner_expert * stride_os_ein
        tl.store(out_ptr, mean_score)


    def _router_forward(
        x: torch.Tensor,
        gate_weights: torch.Tensor,
        up_weights: torch.Tensor,
        expert_ids: torch.Tensor,
        inner_num: int,
        bigger_size: int,
    ) -> torch.Tensor:
        """Wrapper function to launch the Triton kernel."""
        # --- Shape checks and setup ---
        assert x.is_cuda and gate_weights.is_cuda and up_weights.is_cuda and expert_ids.is_cuda
        assert x.is_contiguous() and gate_weights.is_contiguous() and up_weights.is_contiguous()

        B_K, D = x.shape
        E_out, E_in_B_size, _ = gate_weights.shape
        E_in = inner_num
        B_SIZE = bigger_size
        assert E_in_B_size == E_in * B_SIZE

        # --- Output tensor ---
        # Shape: (B*K, E_in)
        out_scores = torch.empty((B_K, E_in), device=x.device, dtype=torch.float32)

        # --- Grid definition ---
        grid = (B_K, E_in)

        # --- Kernel launch ---
        router_forward_kernel[grid](
            x, gate_weights, up_weights, expert_ids, out_scores,
            D, E_out, E_in, B_SIZE,
            x.stride(0), x.stride(1),
            gate_weights.stride(0), gate_weights.stride(1), gate_weights.stride(2),
            up_weights.stride(0), up_weights.stride(1), up_weights.stride(2),
            expert_ids.stride(0),
            out_scores.stride(0), out_scores.stride(1),
        )
        return out_scores

except ImportError:
    print("Triton not found. Falling back to PyTorch implementation.")
    _router_forward = None

class RouterCompoundFast(nn.Module):
    def __init__(self, config, layer_index=-1):
        super().__init__()
        if hasattr(config,"use_mapping") and config.use_mapping:
            dropped_num=config.dropped_num
        else:
            dropped_num=0
        if hasattr(config,"n_routed_experts"):
            self.n_routed_experts = (config.n_routed_experts+dropped_num)//config.inner_num #deepseek
        elif hasattr(config,"num_experts"):
            self.n_routed_experts = (config.num_experts+dropped_num)//config.inner_num #qwen,olmoe
        else:
            raise(KeyError, "num experts not found")  
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.out_gate_weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.inner_num = config.inner_num
        self.acti_num = config.num_experts_per_tok
        self.acti_pattern = config.acti_pattern
        self.routed_scaling_factor = config.routed_scaling_factor
        
        assert len(self.acti_pattern) == self.acti_num 

        self.hidden_size = config.hidden_size
        self.bigger_size = config.bigger_size
                
        self.stacked_in_gate_weights = nn.Parameter(
            torch.empty((self.n_routed_experts, self.bigger_size*self.inner_num, self.gating_dim))
        )
        self.stacked_in_up_weights = nn.Parameter(
            torch.empty((self.n_routed_experts, self.bigger_size*self.inner_num, self.gating_dim))
        )
        
        self.deepseek_style = True if 'deepseek' in config.model_type else False

        # try:
        #     self.use_mapping = config.use_mapping
        # except:
        #     self.use_mapping = False
        # if self.use_mapping:
        #     self.mapping = torch.tensor(config.expert_map[layer_index])
        # else:
        #     self.mapping = torch.ones(self.n_routed_experts)


    def init_weight(self, out_gate: nn.Linear , in_gates_list: nn.ModuleList,):
        self.out_gate_weight = nn.Parameter(out_gate.weight.data) if isinstance(out_gate,nn.Linear) else nn.Parameter(out_gate) #可以直接接受权重
        
        stacked_gate_weights = torch.stack([g.gate.weight for g in in_gates_list])
        stacked_up_weights = torch.stack([g.up.weight for g in in_gates_list])
        assert(stacked_gate_weights.numel()==self.stacked_in_gate_weights.numel())
        assert(stacked_up_weights.numel()==self.stacked_in_up_weights.numel())

        self.stacked_in_gate_weights = nn.Parameter(stacked_gate_weights.reshape(self.stacked_in_gate_weights.shape))
        self.stacked_in_up_weights = nn.Parameter(stacked_up_weights.reshape(self.stacked_in_up_weights.shape))

    def forward_torch(self, x: torch.Tensor, flat_x: torch.Tensor, flat_expert_ids: torch.Tensor) -> torch.Tensor:
        """Original PyTorch implementation for Step 3 for comparison or fallback."""
        batch_gate_weights = self.stacked_in_gate_weights[flat_expert_ids]
        batch_up_weights = self.stacked_in_up_weights[flat_expert_ids]
        
        flat_x_reshaped = flat_x.unsqueeze(-1)
        
        gate_out = torch.bmm(batch_gate_weights, flat_x_reshaped).squeeze(-1)
        up_out = torch.bmm(batch_up_weights, flat_x_reshaped).squeeze(-1)
        
        scores = (up_out * F.silu(gate_out)).abs()
        scores = scores.view(-1, self.inner_num, self.bigger_size)
        all_inner_scores = scores.mean(dim=2)
        return all_inner_scores

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, dim = x.shape
        device = x.device

        # Step 1: Outer expert selection
        logits = F.linear(
            x.type(torch.float32), self.out_gate_weight.type(torch.float32), None
        )
        out_scores = F.softmax(logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(out_scores, self.acti_num, dim=-1)

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        else:
            routing_weights *= self.routed_scaling_factor

        # Step 2: Prepare inputs
        flat_x = x.repeat_interleave(self.acti_num, dim=0)
        flat_expert_ids = selected_experts.reshape(-1)
        flat_weights = routing_weights.reshape(-1)

        # --- Step 3: Inner Expert Scoring (Triton or PyTorch) ---
        if _router_forward is not None and x.is_cuda:
            # 使用 Triton Fused Kernel
            all_inner_scores = _router_forward(
                flat_x,
                self.stacked_in_gate_weights,
                self.stacked_in_up_weights,
                flat_expert_ids,
                self.inner_num,
                self.bigger_size
            )
        else:
            # 使用原始 PyTorch 实现作为后备
            all_inner_scores = self.forward_torch(x, flat_x, flat_expert_ids)
        # all_inner_scores shape: (B*K, E_in)

        # Step 4: Inner top-k routing
        inner_topks = torch.tensor(self.acti_pattern, device=device)[None, :].expand(bs, -1).reshape(-1)
        max_topk = max(self.acti_pattern)#固定4?
        _, topk_indices = torch.topk(all_inner_scores, k=max_topk, dim=-1)

        arange = torch.arange(max_topk, device=device)[None, :]
        mask = arange < inner_topks[:, None]

        flat_expert_ids_expanded = flat_expert_ids[:, None].expand(-1, max_topk)
        selected_inner_ids = flat_expert_ids_expanded * self.inner_num + topk_indices

        total_activated_experts = sum(self.acti_pattern)
        final_ids = torch.masked_select(selected_inner_ids, mask).view(bs, total_activated_experts)

        expanded_weights = flat_weights[:, None].expand(-1, max_topk)
        final_weights = torch.masked_select(expanded_weights, mask).view(bs, total_activated_experts)
        
        # breakpoint()
        if self.deepseek_style:
            return final_ids, final_weights, None
        else:
            return final_weights, final_ids


class NewCompoundMoE(nn.Module):
    def __init__(self, gate, experts_list, num_experts, top_k, norm_topk_prob, shared_expert=None):
        super().__init__()
        self.experts:nn.ModuleList = experts_list
        self.gate:RouterCompound = gate
        self.gate.norm_topk_prob = norm_topk_prob #是否给routing weights做normalize，olmoe为false，mixtral为true
        self.num_experts = num_experts
        self.shared_expert = shared_expert 
        self.top_k = top_k #没用
        self.name = 'normal'
        self.opti = False 
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # try:
        if self.opti:
            return self.opti_forward(hidden_states)
        # except:
        #     pass
        if self.name =='deepseek':
            return self.dpsk_forward(hidden_states)
        identity = hidden_states
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        routing_weights, selected_experts = self.gate(hidden_states)

        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be selected
        # expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            try:
                idx, top_x = torch.where(selected_experts == expert_idx) #返回行索引（batch中的哪个） 和 列索引（top几的expert）
            except:
                continue #说明where一个都没查找到，那说明这个batch下的token全没用到这个expert

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[idx].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[idx, top_x, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, idx, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        # print(final_hidden_states)
        # exit()
        if self.shared_expert is not None:
            shared_hidden_states = self.shared_expert(identity)
            final_hidden_states += shared_hidden_states
        
        return final_hidden_states, routing_weights
    
    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts))) # (T, E)
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0) # E
        idxs = topk_ids.view(-1).argsort() # (T*K)
        # topk_ids 展平成一个一维长向量，长度为 总Token数 * K 。这个向量包含了所有 token 的所有专家分配
        # argsort对这个展平的向量进行升序排序，并返回排序后元素对应的原始索引
        # 这个索引可以用来重新排列 token，使得所有被分配给同一个专家的 token 在物理上是连续的
        sorted_tokens = x[idxs // topk_ids.shape[1]] # idxs // topk_ids.shape[1] 将 idxs 转换为原始 token 的索引
        sorted_tokens_shape = sorted_tokens.shape # (T*K, D)
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0) # (T*K, D)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs # 还原了 token 被排序之前的顺序
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        # print(final_out)
        # exit()
        return final_out

    def dpsk_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        topk_weight, topk_idx = self.gate(hidden_states)
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.shared_expert is not None:
            y = y + self.shared_expert(identity)
        return y,topk_weight
    
    def opti_forward(self, hidden_states: torch.Tensor) -> torch.Tensor: #暂时只写了deepseek版本
        identity = hidden_states
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1]) # (T, D)

        out = self.gate.out_gate(hidden_states)
        out_scores = F.softmax(out, dim=-1, dtype=torch.float32)  # (T, E)
        topk_weight, topk_idx  = torch.topk(out_scores, self.gate.acti_num, dim=-1)  # (T, K)

        if self.gate.norm_topk_prob: 
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        else:
            topk_weight *= self.gate.routed_scaling_factor

        topk_pattern = torch.tensor(self.gate.acti_pattern).unsqueeze(0) # (1, K)
        topk_pattern = topk_pattern.repeat(topk_weight.shape[0], 1) # (T, K)

        # ---moe_infer---

        cnts = topk_idx.new_zeros((topk_idx.shape[0], len(self.experts))) # (T, E)
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0) # E
        idxs = topk_idx.view(-1).argsort() # (T*K)
        # topk_ids 展平成一个一维长向量，长度为 总Token数 * K 。这个向量包含了所有 token 的所有专家分配
        # argsort对这个展平的向量进行升序排序，并返回排序后元素对应的原始索引
        # 这个索引可以用来重新排列 token，使得所有被分配给同一个专家的 token 在物理上是连续的
        sorted_tokens = hidden_states[idxs // topk_idx.shape[1]] # idxs // topk_ids.shape[1] 将 idxs 转换为原始 token 的索引
        sorted_tokens_shape = sorted_tokens.shape # (T*K, D)
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        sorted_patterns = topk_pattern.view(-1)[idxs.cpu()] # (T*K)

        def acti(model,x):
            return model.act_fn(model.gate_proj(x))*model.up_proj(x)

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            experts = self.experts[i*self.gate.inner_num  : (i+1)*self.gate.inner_num]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx] # (Selected, D)
            pattern_for_this_expert = sorted_patterns[start_idx:end_idx].to(device=tokens_for_this_expert.device) # (Selected)

            expert_acti = torch.stack([acti(experts[i],tokens_for_this_expert) for i in range(self.gate.inner_num)], dim=0) # subE * (Selected, I)
            expert_acti_norm = torch.linalg.norm(expert_acti, ord=1, dim=(2)).T # (subE, Selected).T
            k_max = torch.max(pattern_for_this_expert)
            _, topk_indices = torch.topk(expert_acti_norm, k=k_max, dim=-1) # Selected, inner_K_max
            arange_k = torch.arange(k_max, device=tokens_for_this_expert.device)
            mask = arange_k < pattern_for_this_expert.unsqueeze(1)
            masked_indices = torch.where(mask, topk_indices, torch.tensor(-1, device=tokens_for_this_expert.device)) # Selected, inner_K_max

            final = hidden_states.new_zeros((num_tokens,hidden_states.shape[-1]))  # (Selected, D)
            for s in range(self.gate.inner_num):
                try:
                    idx, top_x = torch.where(masked_indices == s)
                except:
                    continue
                current_acti = expert_acti[s][idx]
                current_out = experts[s].down_proj(current_acti)
                final.index_add_(0, idx, current_out)

            outputs.append(final) # (Selected, D)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0) # (T*K, D)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs # 还原了 token 被排序之前的顺序
        final_out = (
            new_x.view(*topk_idx.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        
        # ---end moe_infer---

        y = final_out.view(*orig_shape)
        if self.shared_expert is not None:
            y = y + self.shared_expert(identity)
        return y,topk_weight





class ZeroLayer(nn.Module):
    """
    无论输入是什么，都输出一个形状、数据类型和设备都与输入相同的全零张量。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)