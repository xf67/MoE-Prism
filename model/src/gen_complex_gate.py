# 接受linear gate模型，生成complex gate

from transformers import PreTrainedModel
import torch
from torch import nn
from tqdm import tqdm
import os
import numpy as np
from utils_new import MLPWrapper
from custom_models import RouterBigger,LlamaMLP,MoE
from custom_models import RouterCompoundFast,NewCompoundMoE
import logging
logger = logging.getLogger('log') 
DEVICE = 'cuda'

class DummyClassForPassingItems():
    def __init__(self, ):
        pass

@torch.no_grad()
def run_get_complex(model:PreTrainedModel,data_list,config,model_config):
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
        
        in_gates_list=[]
        for idx in tqdm(range(num_experts),desc=f"Enumerating experts"):
            if not_moe:
                break
            
            original_expert = moe_layer.experts[idx]
            moe_layer.experts[idx] = original_expert.module # 拆掉wrapper放回去
            neurons_per_expert = original_expert.acti_buffer[0].shape[1]
            # activate_score= torch.concat(original_expert.acti_buffer,dim=0).cuda().abs() 
            gpu_buffers = [b.cuda(non_blocking=True) for b in original_expert.acti_buffer]
            activate_score = torch.concat(gpu_buffers, dim=0).abs()

            if config['gate']['relation']['s']==0:
                k_gate = config['gate']['relation']['s_true']
            else:
                k_gate = neurons_per_expert // config['gate']['relation']['s']
            
            act_k = neurons_per_expert*config['partition']['k']//config['partition']['n'] #这里用k/n带了一下act_k
            _,activation_topk = torch.topk(activate_score,k=act_k,dim=-1)
            binary_mask = torch.zeros_like(activate_score)
            binary_mask.scatter_(dim=-1, index=activation_topk, value=1)
            del activation_topk
            cross_activate_full = (binary_mask.cuda().T @ binary_mask.cuda())
            sums = torch.sum(cross_activate_full,dim=1)
            # medoid_idx_within_cluster = np.argpartition(sums, -k_gate)[-k_gate:] #相当于topk
            _,medoid_idx_within_cluster = torch.topk(sums, k_gate)

            gate_proj = moe_layer.experts[idx].w1 if name=='mixtral' else moe_layer.experts[idx].gate_proj
            up_proj = moe_layer.experts[idx].w3 if name=='mixtral' else moe_layer.experts[idx].up_proj

            core_gate=gate_proj.weight.data[medoid_idx_within_cluster, :]
            core_up=up_proj.weight.data[medoid_idx_within_cluster, :]

            router = DummyClassForPassingItems()
            router.gate = DummyClassForPassingItems()
            router.gate.weight = core_gate
            router.up = DummyClassForPassingItems()
            router.up.weight = core_up
            in_gates_list.append(router)

        if not_moe:
            continue
        nnn = config['partition']['n']
        out_gate_weight = moe_layer.gate.weight.data[0::nnn]
        new_gate=RouterCompoundFast(model_config)
        new_gate.init_weight(out_gate_weight,in_gates_list)
        moe_layer.gate = new_gate
        if name=='mixtral':
            decoder_layer.block_sparse_moe = moe_layer
        else:
            decoder_layer.mlp = moe_layer

        torch.cuda.empty_cache()
        decoder_layer = decoder_layer.to(config['offload_device']) #卸载到CPU上
    model.config.use_cache = True
    