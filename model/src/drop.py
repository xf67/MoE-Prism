import os
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from utils_new import GateWrapper
import logging
logger = logging.getLogger('log') 
DEVICE = 'cuda'

@torch.no_grad()
def run_dropping(model:PreTrainedModel,data_list,config):
    
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
        topk=layers[1].block_sparse_moe.top_k
    elif name=='olmoe' or name=='qwen':
        num_experts=layers[1].mlp.num_experts
        topk=layers[1].mlp.top_k
    elif name=='deepseek':
        num_experts=layers[1].mlp.config.n_routed_experts
        topk=layers[1].mlp.gate.top_k
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
            original_gate = moe_layer.gate
            moe_layer.gate = GateWrapper(original_gate,top_k=topk)

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

        if not_moe:   
            continue #不是moe算完就可以走了

        activated_experts = torch.cat(moe_layer.gate.gate_acti_buffer).flatten().numpy() # 该层全部激活的expert的idx
        moe_layer.gate = moe_layer.gate.module_wrapped  # 恢复原来的gate
        
        # print(activated_experts)

        activated_rate = np.bincount(activated_experts) # 长度是expert的总数目，qwen是240

        if config['debug']['save_activations_for_drop']:
            SaveFolder=config['debug']['save_folder']
            SaveFileName=config['debug']['save_file_name']
            if not os.path.exists(SaveFolder):
                os.mkdir(SaveFolder)
            with open(f'{SaveFolder}/{SaveFileName}', 'a', encoding='utf-8') as f:
                f.write("\n")
                f.write(np.array2string(activated_rate, separator=', '))
            logger.info(f"Saved activation rate for layer{layer_index}")
        
        drop_num = config['drop']['list'][layer_index] if layer_index<len(config['drop']['list']) else config['drop']['list'][-1]
        assert(drop_num<num_experts, f"drop too many experts: drop_num {drop_num} > num_experts {num_experts}")

        dropped_experts = np.argsort(activated_rate)[:drop_num]

        for dropped_idx in dropped_experts:
            if config['io']['model_type']=='mixtral:':
                moe_layer.experts[dropped_idx].down_proj.weight.data = torch.zeros_like(moe_layer.experts[dropped_idx].down_proj.weight.data)
            else:
                moe_layer.experts[dropped_idx].w2.weight.data = torch.zeros_like(moe_layer.experts[dropped_idx].w2.weight.data)


