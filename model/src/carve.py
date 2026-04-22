from transformers import PreTrainedModel
import torch
from torch import nn
from tqdm import tqdm
import os
import numpy as np
from utils_new import MLPWrapper,wrap_metis,wrap_kmeans,wrap_kmeans_simple,wrap_sa,wrap_rand
from custom_models import RouterBigger,LlamaMLP,MoE
from custom_models import RouterCompoundFast,NewCompoundMoE
import logging
logger = logging.getLogger('log') 
DEVICE = 'cuda'

@torch.no_grad()
def run_carving(model:PreTrainedModel,data_list,config,model_config):

    dtype = next(iter(model.parameters())).dtype
    model.config.use_cache = False
    inps = torch.zeros(
            (len(data_list), config['bsz'], min(model.config.max_position_embeddings,4096), model.config.hidden_size), dtype=dtype, device=config['offload_device']
        ) #22GB，85*1*32768*4096 *2(half精度), for mixtral
    logger.info(f"Get dataloader done, data shape: {inps.shape}")

    name = config['io']['model_type']
    cache = {'i': 0, 'attention_mask': [], 'position_ids': [], 'past_key_value': [], 'position_embeddings': []}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'].append(kwargs['attention_mask'])
            cache['position_ids'].append(kwargs['position_ids'])
            cache['past_key_value'].append(kwargs['past_key_value'])
            if name=='olmoe' or name=='qwen' or name=='mixtral':
                cache['position_embeddings'].append(kwargs['position_embeddings'])
            raise ValueError
        
    layers=model.model.layers
    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in data_list:
            try:
                model(batch.to(config['offload_device']))
            except ValueError:
                pass
    layers[0] = layers[0].module
    #全部在CPU(offload_device)上做，不需要涉及GPU，因为没算到attention层
    logger.info("Prepare input calibration data done.")

    def try_to(something,dev): #为了解决有的东西是None
        try:
            return something.to(dev)
        except:
            # print(f"Try failed {something}")
            return something

    # inps 是 85x1x32768x4096, 当bsz=1,len=32768
    # inps 是 129, 1, 4096, 409  当bsz=1,len=4096. len太大了容易爆显存
    # inps一次性传完85个会爆显存
    hidden_states_batch=[None for _ in range(inps.shape[0])]
    attention_mask_batch=[None for _ in range(inps.shape[0])]
    position_ids_batch=[None for _ in range(inps.shape[0])]
    past_key_value_batch=[None for _ in range(inps.shape[0])]
    position_embeddings_batch=[None for _ in range(inps.shape[0])]

    #下面开始在GPU上做，每次做完之后把这个layer从GPU上卸载，然后加载下一个层
    layer_index=-1
    num_experts=0
    if name=='mixtral':
        num_experts=layers[1].block_sparse_moe.num_experts
    elif name=='olmoe' or name=='qwen':
        num_experts=layers[1].mlp.num_experts
    elif name=='deepseek':
        num_experts=layers[1].mlp.config.n_routed_experts

    for decoder_layer in layers:
        layer_index+=1
        if layer_index>config['early_stop']:
            logger.info(f"Early stop at layer{layer_index-1}")
            break
        logger.info(f"Processing layer{layer_index}")
        decoder_layer = try_to(decoder_layer,DEVICE).to(dtype=torch.bfloat16)
        moe_layer = decoder_layer.block_sparse_moe if name=='mixtral' else decoder_layer.mlp

        not_moe=False
        g = getattr(moe_layer, 'gate', None)
        if not g:
            not_moe = True

        if not not_moe:   
            for i in range(len(moe_layer.experts)):
                original_expert = moe_layer.experts[i]
                moe_layer.experts[i] = MLPWrapper(original_expert)

        for i in tqdm(range(inps.shape[0]), desc = 'Running through batches'):
            if hidden_states_batch[i]==None:
                inp=inps[i]
                hidden_states_batch[i] = try_to(inp,DEVICE)
                attention_mask_batch[i] = try_to(cache['attention_mask'][i],DEVICE)
                position_ids_batch[i] = try_to(cache['position_ids'][i],DEVICE)
                past_key_value_batch[i] = try_to(cache['past_key_value'][i],DEVICE)
                if name=='olmoe' or name=='qwen' or name=='mixtral':
                    position_embeddings_batch[i] = (cache['position_embeddings'][i][0].to(DEVICE).to(dtype=torch.bfloat16),cache['position_embeddings'][i][1].to(DEVICE).to(dtype=torch.bfloat16))

            hidden_states=hidden_states_batch[i].to(DEVICE).to(dtype=torch.bfloat16)

            kwargs = {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask_batch[i],
                "position_ids": position_ids_batch[i],
                "past_key_value": past_key_value_batch[i],
                "output_attentions": False,
                "use_cache": False,
            }
            if name!='deepseek':
                kwargs["cache_position"] = None
            if name=='olmoe' or name=='qwen' or name=='mixtral':
                kwargs['position_embeddings'] = position_embeddings_batch[i]

            hidden_states = decoder_layer(**kwargs)[0]

            hidden_states_batch[i] = hidden_states.float().to(config['offload_device'])
            # print(hidden_states_batch[i].shape) # 1 4096 2048
            torch.cuda.empty_cache()

        # 一层跑完

        if config['debug']['save_activations'] and not not_moe:
            SaveFolder=config['debug']['save_folder']
            if not os.path.exists(SaveFolder):
                os.mkdir(SaveFolder)
            for idx,expert in enumerate(moe_layer.experts):
                torch.save(expert.acti_buffer,f"{SaveFolder}/Acti_{layer_index}_{idx}.pth")
                torch.save(expert.input_buffer,f"{SaveFolder}/Input_{layer_index}_{idx}.pth")
            logger.info(f"Saved activation rate for layer{layer_index}")
            exit()


        all_experts=nn.ModuleList()
        for idx in tqdm(range(num_experts),desc=f"Enumerating experts"):
            if not_moe:
                break
                
            expert_layer:MLPWrapper = moe_layer.experts[idx]
            if config['partition']['method']=='kmeans' and not  config['partition']['kmeans']['simple']:
                if name=='mixtral':
                    Wg=expert_layer.module.w1.weight.data
                    Wu=expert_layer.module.w3.weight.data
                else:
                    Wg=expert_layer.module.gate_proj.weight.data # Intermediate_size * Hidden_size
                    Wu=expert_layer.module.up_proj.weight.data # Intermediate_size * Hidden_size

            # input_state=expert_layer.input_buffer
            
            neuron_size = expert_layer.acti_buffer[0].shape[1]
            neurons_per_expert = neuron_size // config['partition']['n']
            remaining_indices = set(range(neuron_size))
            act_k = neurons_per_expert*config['partition']['k']

            # logger.debug(f"Tokens in layer{layer_index} expert{idx}: {[iss.shape[0] for iss in input_state]}")
            
            # input_state = torch.concat(input_state,dim=0).cuda() # Tokens*Hidden_size
            activate_score= torch.concat(expert_layer.acti_buffer,dim=0).cuda().abs() # Tokens*Intermediate_size
            if config['partition']['shared']+config['partition']['drop'] >0  or config['gate']['method']=='relation':
                _,activation_topk = torch.topk(activate_score,k=act_k,dim=-1)
                binary_mask = torch.zeros_like(activate_score)
                binary_mask.scatter_(dim=-1, index=activation_topk, value=1)
                del activation_topk
                binary_mask_sum=binary_mask.sum(dim=0)

            # logger.debug(f"Tokens in layer{layer_index} expert{idx}: {input_state.shape[0]}")

            if config['partition']['shared']+config['partition']['drop'] >0:
            # shared
                top_actiavted = torch.topk(binary_mask_sum,k=neurons_per_expert*config['partition']['shared'])[1]
                remaining_indices -= set(top_actiavted.tolist())
                # drop
                less_activated = torch.topk(binary_mask_sum,k=neurons_per_expert*config['partition']['drop'],largest=False)[1]
                remaining_indices -= set(less_activated.tolist())
            else:
                top_actiavted=torch.empty(0)
                less_activated=torch.empty(0)
            remaining_indices_list = list(remaining_indices)

            
            if config['partition']['method']=='metis' or config['gate']['method']=='relation':
                cross_activate_full = (binary_mask.cuda().T @ binary_mask.cuda()).to(config['offload_device'])
                cross_activate = cross_activate_full[remaining_indices_list][:, remaining_indices_list]

            if config['partition']['method']=='kmeans' and not  config['partition']['kmeans']['simple']:
                binary_mask_remain = binary_mask[:,remaining_indices_list]
                Wg_remain = Wg[remaining_indices_list] # Remain_intermediate_size * Hidden_size
                Wu_remain = Wu[remaining_indices_list] # Remain_intermediate_size * Hidden_size
            activate_score_remain = activate_score[:,remaining_indices_list] # Tokens * Remain_intermediate_size
            # input_state 不用动

            if config['partition']['method']=='metis':
                partitionsK=wrap_metis(cross_activate.cpu(),config['partition']['metis']['threshold'],config['partition']['n']) #metis只能cpu，metis也很久没维护了
            elif config['partition']['method']=='kmeans':
                if config['partition']['kmeans']['simple']:
                    partitionsK=wrap_kmeans_simple(binary_mask_remain,config['partition']['n']-config['partition']['shared']-config['partition']['drop'],neurons_per_expert,config['partition']['kmeans']['iter'])
                else:
                    _, top_indices = torch.topk(activate_score_remain, config['partition']['n']) # 不是simple是之前的代码copy过来的，很久没维护了，不清楚是不是必须cpu，目前先用simple的再说
                    selected_index = [remaining_indices_list[idx] for idx in top_indices]
                    centroids = binary_mask_remain[:, selected_index].clone()
                    W_=torch.concat([Wg,Wu],dim=-1)
                    W_centroid=W_[centroids]
                    partitionsK=wrap_kmeans(binary_mask_remain,Wg_remain,Wu_remain,W_centroid,cross_activate.shape[0],config['partition']['n']-config['partition']['shared']-config['partition']['drop'],config['partition']['kmeans']['iter'],config['partition']['kmeans']['aplha'])
            elif config['partition']['method']=='sa':
                partitionsK=wrap_sa(activate_score_remain,config['partition']['n']-config['partition']['shared']-config['partition']['drop'],config['partition']['k'],config['partition']['sa']['iter'],config['partition']['sa']['initial_temp'],config['partition']['sa']['cooling_rate'])
            elif config['partition']['method']=='random':
                partitionsK=wrap_rand(len(remaining_indices_list),config['partition']['n']-config['partition']['shared']-config['partition']['drop'],)
            else:
                logger.error("Error: Partition method")
                raise(NotImplementedError)

            partitions={idx: [] for idx in range(config['partition']['n']-config['partition']['drop'])}
            partitions[0] = top_actiavted.tolist() #第一个expert的第一个分组是always_activate的神经元


            for i in range(config['partition']['n']-config['partition']['shared']-config['partition']['drop']):
                partitions[i+config['partition']['shared']] = [remaining_indices_list[s] for s in partitionsK[i]]

            #取聚类中心
            if config['gate']['method']=='relation':
                if config['gate']['relation']['s']==0:
                    k_gate = config['gate']['relation']['s_true']
                else:
                    k_gate = neurons_per_expert // config['gate']['relation']['s']
                cluster_medoids = []
                for Sidx in range(config['partition']['shared'],config['partition']['n']-config['partition']['drop']):
                    idx_list=list(partitions[Sidx])

                    cluster_vectors = cross_activate_full[idx_list][:,idx_list].numpy()
                    sums = np.sum(cluster_vectors,axis=1)

                    medoid_idx_within_cluster = np.argpartition(sums, -k_gate)[-k_gate:] #相当于topk
                    medoid_idx = [ idx_list[i] for i in medoid_idx_within_cluster ]
                    cluster_medoids.append(medoid_idx)
            

                #重建expert
                original_expert = moe_layer.experts[idx]
                moe_layer.experts[idx] = original_expert.module
                experts = nn.ModuleList() #重建的experts
                router = RouterBigger(model.config.hidden_size,config['partition']['n']-config['partition']['shared']-config['partition']['drop'],config['partition']['k'],k_gate)
                core_gate=[]
                core_up=[]
                if config['partition']['shared']==0:
                    experts.append(None)
                for Sidx in range(config['partition']['n']-config['partition']['drop']):
                    if Sidx!=0 and Sidx<config['partition']['shared']:
                        continue
                    expert_indices = partitions[Sidx]
                    gate_proj = moe_layer.experts[idx].w1 if name=='mixtral' else moe_layer.experts[idx].gate_proj
                    up_proj = moe_layer.experts[idx].w3 if name=='mixtral' else moe_layer.experts[idx].up_proj
                    down_proj =  moe_layer.experts[idx].w2 if name=='mixtral' else moe_layer.experts[idx].down_proj
                    gate_dim = gate_proj.weight.shape[1] # same as model.config.hidden_size

                    # expert_mlp = LlamaMLP(gate_dim, len(expert_indices)).to(config['offload_device'])
                    thisMLP = type(moe_layer.experts[idx])
                    if name=='deepseek' or name=='qwen':
                        expert_mlp = thisMLP(model_config,intermediate_size=model_config.moe_intermediate_size)
                    elif name=='olmoe' or name=='mixtral':
                        expert_mlp = thisMLP(model_config)
                    else:
                        raise(NotImplementedError)
                    expert_mlp.to(config['offload_device'])
                    
                    # Initialize weights from the corresponding neurons in original projections
                    with torch.no_grad():
                        # expert_mlp.gate_proj.weight.data = gate_proj.weight.data[expert_indices, :]
                        # expert_mlp.up_proj.weight.data = up_proj.weight.data[expert_indices, :]
                        # expert_mlp.down_proj.weight.data = down_proj.weight.data[:, expert_indices]
                        if name != 'mixtral':
                            expert_mlp.gate_proj.weight.data = gate_proj.weight.data[expert_indices, :]
                            expert_mlp.up_proj.weight.data = up_proj.weight.data[expert_indices, :]
                            expert_mlp.down_proj.weight.data = down_proj.weight.data[:, expert_indices]
                        else:
                            expert_mlp.w1.weight.data = gate_proj.weight.data[expert_indices, :]
                            expert_mlp.w3.weight.data = up_proj.weight.data[expert_indices, :]
                            expert_mlp.w2.weight.data = down_proj.weight.data[:, expert_indices]
                    experts.append(expert_mlp)

                    # Initialize core_neuron
                    if Sidx>=config['partition']['shared']:
                        core_gate.append(gate_proj.weight.data[cluster_medoids[Sidx-config['partition']['shared']], :])
                        core_up.append(up_proj.weight.data[cluster_medoids[Sidx-config['partition']['shared']], :])
                core_gate=torch.concat(core_gate,dim=0)
                core_up=torch.concat(core_up,dim=0)
                router.gate.weight.data = core_gate #原CMoE实现这里还做了一步normalize，感觉可能没必要
                router.up.weight.data = core_up

                moe = MoE(model.config.hidden_size, model.config.intermediate_size//config['partition']['n'], config['partition']['n']-config['partition']['drop'], config['partition']['shared'], config['partition']['k'], k_gate) 
                moe.gate = router
                moe.experts = experts[1:]
                moe.shared_experts = experts[0]
                moe.cus_training = False
                
                moe_layer.experts[idx]=moe # 在这里把原来的expert换成新构建的moe,到这里为止是双层的结构，后面处理成单层

            elif config['gate']['method']=='linear':
                original_expert = moe_layer.experts[idx]
                moe_layer.experts[idx] = original_expert.module
                for Sidx in range(config['partition']['n']-config['partition']['drop']):
                    if Sidx!=0 and Sidx<config['partition']['shared']:
                        continue
                    expert_indices = partitions[Sidx]

                    thisMLP = type(moe_layer.experts[idx])
                    if name=='deepseek' or name=='qwen':
                        expert_mlp = thisMLP(model_config,intermediate_size=model_config.moe_intermediate_size)
                    elif name=='olmoe' or name=='mixtral':
                        expert_mlp = thisMLP(model_config)
                    else:
                        raise(NotImplementedError)
                    
                    # Initialize weights from the corresponding neurons in original projections
                    with torch.no_grad():
                        if name != 'mixtral':
                            expert_mlp.gate_proj.weight.data = moe_layer.experts[idx].gate_proj.weight.data[expert_indices, :]
                            expert_mlp.up_proj.weight.data = moe_layer.experts[idx].up_proj.weight.data[expert_indices, :]
                            expert_mlp.down_proj.weight.data = moe_layer.experts[idx].down_proj.weight.data[:, expert_indices]
                        else:
                            expert_mlp.w1.weight.data = moe_layer.experts[idx].w1.weight.data[expert_indices, :]
                            expert_mlp.w3.weight.data = moe_layer.experts[idx].w3.weight.data[expert_indices, :]
                            expert_mlp.w2.weight.data = moe_layer.experts[idx].w2.weight.data[:, expert_indices]
                    torch.cuda.empty_cache()
                    all_experts.append(expert_mlp.to(config['offload_device']))
            else:
                logger.error("Error: Gate selection method")
                raise(NotImplementedError)
            torch.cuda.empty_cache()
        
        if not not_moe:
            if config['gate']['method']=='linear':
                thisMoE = type(moe_layer)
                moe_new = thisMoE(model_config)
                moe_new.experts = all_experts
                moe_new.gate.weight.data = moe_layer.gate.weight.data.repeat_interleave(config['partition']['n'],dim=0)
                if hasattr(moe_layer,'shared_experts'):
                    moe_new.shared_experts = moe_layer.shared_experts
                if hasattr(moe_layer,'shared_expert_gate'):
                    moe_new.shared_expert_gate = moe_layer.shared_expert_gate
                if hasattr(moe_layer,'shared_expert'):
                    moe_new.shared_expert = moe_layer.shared_expert
                if name=='mixtral':
                    decoder_layer.block_sparse_moe = moe_new
                else:
                    decoder_layer.mlp = moe_new
            elif config['gate']['method']=='relation': #把双层gate变成单层
                out_gate = moe_layer.gate
                if name=='deepseek':
                    out_gate.early_output=True
                in_gates_list=nn.ModuleList()
                experts_list=nn.ModuleList()
                for out_expert in moe_layer.experts: #这是原网络的一个expert
                    in_gates_list.append(out_expert.gate) #一共有expert_num个
                    experts_list.extend(out_expert.experts) #一共有expert_num*sub_expert_num个
                new_gate=RouterCompoundFast(model_config)
                new_gate.init_weight(out_gate,in_gates_list)
                moe_layer.experts = experts_list
                moe_layer.gate = new_gate
                if name=='mixtral':
                    decoder_layer.block_sparse_moe = moe_layer
                else:
                    decoder_layer.mlp = moe_layer

        torch.cuda.empty_cache()
        decoder_layer = decoder_layer.to(config['offload_device']) #卸载到CPU上
    model.config.use_cache = True
    


