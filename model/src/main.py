import sys
import os
import yaml
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from logger import setup_logger
from carve import run_carving
from drop import run_dropping
from utils_new import get_wikitext2
from save_utils import save_full,save_model_config
from gen_complex_gate import run_get_complex

def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Error: Can not find '{config_path}' ")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error: Loading '{config_path}' : {e}")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="ok")
    parser.add_argument("-cfg", "--config", type=str, default='./cfg.yaml', help="YAML config")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        print("Error: Getting config")
        exit()

    config.setdefault('offload_device', 'cpu')

    logger = setup_logger(log_dir = f"./logs/{os.path.splitext(os.path.basename(args.config))[0]}.txt")
    
    tokenizer = AutoTokenizer.from_pretrained(config['io']['tokenizer_from'], trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(config['io']['model_from'],
                                                trust_remote_code=True, 
                                                device_map=config['offload_device'],
                                                torch_dtype=torch.bfloat16
                                            )
    
    calibration_set, _ = get_wikitext2(None, None, min(model.config.max_position_embeddings,4096), tokenizer, bsz = config['bsz'], max_batch_num=128) 

    t1=time.time()
    if config['stage']==1:
        model_config=save_model_config(config, args.config)
        run_carving(model, calibration_set, config, model_config)
        save_full(model, config, args.config)

        # sss=config['io']['model_to']
        # if not os.path.exists(sss):
        #     os.makedirs(sss)
        # torch.save(model.to(torch.bfloat16),f'{sss}/model.pth')

    elif config['stage']==2:
        mapping = run_dropping(model, calibration_set, config)
        save_full(model, config, args.config, mapping)

    elif config['stage']==1.5: # gen complex gate from linear gate carved model
        model_config=save_model_config(config, args.config)
        run_get_complex(model, calibration_set, config, model_config)
        save_full(model, config, args.config)


    else:
        print("Error: Config stage should be 1 or 2")
        exit()

    t2=time.time()
    print(f"Time in total: {t2-t1}")
    logger.info(f"Time in total: {t2-t1}")
