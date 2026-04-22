
import os
import shutil
import types
import json


def add_suffix_to_filename(file_path,add_what):

    directory, full_filename = os.path.split(file_path)
    filename_stem, extension = os.path.splitext(full_filename)
    new_filename_stem = filename_stem + add_what
    new_full_filename = new_filename_stem + extension
    new_file_path = os.path.join(directory, new_full_filename)

    return new_file_path

def dict_to_namespace(data):
    """递归地将字典转换为 SimpleNamespace 对象。"""
    if not isinstance(data, dict):
        return data
    
    # 创建一个 SimpleNamespace 对象
    ns = types.SimpleNamespace()
    
    for key, value in data.items():
        # 如果值是字典，递归转换
        if isinstance(value, dict):
            setattr(ns, key, dict_to_namespace(value))
        # 如果值是列表，遍历列表并对其中的字典进行递归转换
        elif isinstance(value, list):
            setattr(ns, key, [dict_to_namespace(item) for item in value])
        # 其他类型的值直接设置
        else:
            setattr(ns, key, value)
            
    return ns



def copy_auto_map_sources(orig_model_dir, save_dir):
    config_path = os.path.join(orig_model_dir, "config.json")
    # 读取 config.json
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    auto_map = config_data.get("auto_map", {})
    if not auto_map:
        print("config.json 中没有 auto_map 字段，跳过复制源码")
        return

    # 提取所有需要复制的源码文件名（去掉类名后缀，取第一个点前部分）
    source_files = set()
    for val in auto_map.values():
        # val 格式形如 "modeling_deepseek.DeepseekV3ForCausalLM"
        filename = val.split(".")[0] + ".py"
        source_files.add(filename)

    print("需要复制的源码文件：", source_files)

    # 复制文件
    for filename in source_files:
        src_file = os.path.join(orig_model_dir, filename)
        dst_file = os.path.join(save_dir, filename)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
            print(f"复制 {filename} 到 {save_dir}")
        else:
            print(f"警告：{filename} 在原目录未找到，跳过复制")


def copy_model_files_without_weights(source_dir: str, destination_dir: str):
    """
    从源目录复制所有文件到目标目录，但排除模型权重文件。

    这个函数会跳过以下后缀的文件：
    - .safetensors
    - .bin

    Args:
        source_dir (str): 包含完整模型文件的源目录路径。
        destination_dir (str): 要将文件复制到的目标目录路径。
    """
    # 1. 检查源目录是否存在
    if not os.path.isdir(source_dir):
        print(f"Error: Source folder '{source_dir}' does not exists")
        return

    # 2. 创建目标目录（如果它还不存在的话）
    # exist_ok=True 表示如果目录已存在，则不会报错
    os.makedirs(destination_dir, exist_ok=True)
    print(f"Destination folder '{destination_dir}' ok")

    # 定义要排除的文件后缀名
    excluded_extensions = (".safetensors", ".bin")

    # 3. 遍历源目录中的所有文件和文件夹
    copied_files_count = 0
    skipped_files_count = 0
    for filename in os.listdir(source_dir):
        # 构建完整的文件路径
        source_path = os.path.join(source_dir, filename)

        # 只处理文件，忽略子目录（Hugging Face 模型目录通常是扁平的）
        if os.path.isfile(source_path):
            # 4. 检查文件后缀是否在排除列表中
            if filename.endswith(excluded_extensions):
                print(f"  - [Passed] Weight: {filename}")
                skipped_files_count += 1
            else:
                # 5. 如果不是权重文件，则复制它
                destination_path = os.path.join(destination_dir, filename)
                print(f"  - [Copied] Config: {filename}")
                shutil.copy2(source_path, destination_path) # copy2 会同时复制元数据
                copied_files_count += 1

    print("\nCopying Done")
    print(f"Totally {copied_files_count} files copied, {skipped_files_count} files passed.")



def save_model_config(config,yaml_path,mapping=None):
    save_dir=config['io']['model_to']
    os.makedirs(save_dir, exist_ok=True)
    
    # copy_auto_map_sources(config['io']['model_from'], save_dir)
    for f in config['io']['addition']:
        shutil.copy(f, save_dir)
    shutil.copy(yaml_path, save_dir)

    config_from_path = os.path.join(config['io']['model_from'], "config.json")
    with open(config_from_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    if config['stage']==1:

        if config['io']['model_type'] == 'olmoe':
            config_data['num_experts'] *= config['partition']['n']
            config_data['num_experts_per_tok'] *= config['partition']['n']
            config_data['intermediate_size'] //= config['partition']['n']
            config_data['routed_scaling_factor'] = config['partition']['n']
        elif config['io']['model_type'] == 'deepseek':
            config_data['n_routed_experts'] *= config['partition']['n']
            config_data['num_experts_per_tok'] *= config['partition']['n']
            config_data['n_shared_experts'] *= config['partition']['n']
            config_data['routed_scaling_factor'] *= config['partition']['n'] #用*=因为它原来就有这个参数
            config_data['moe_intermediate_size'] //= config['partition']['n']
        elif config['io']['model_type'] == 'mixtral':
            config_data['num_local_experts'] *= config['partition']['n']
            config_data['num_experts_per_tok'] *= config['partition']['n']
            config_data['intermediate_size'] //= config['partition']['n']
            config_data['routed_scaling_factor'] = config['partition']['n']
        elif config['io']['model_type'] == 'qwen':
            config_data['num_experts'] *= config['partition']['n']
            config_data['num_experts_per_tok'] *= config['partition']['n']
            # config_data['intermediate_size'] //= config['partition']['n']
            config_data['moe_intermediate_size'] //= config['partition']['n']
            config_data['routed_scaling_factor'] = config['partition']['n']
        else:
            raise(NotImplementedError)

        config_data["use_mask"] = False
        config_data["use_list"] = False

    if config['gate']['method'] == 'relation' and (config['stage']==1 or config['stage']==1.5):
        config_data["inner_num"] = config['partition']['n']
        config_data["acti_pattern"] = config['gate']['list']
        if config['gate']['relation']['s']==0:
            config_data["bigger_size"]  = config['gate']['relation']['s_true']
        else:
            ss=config['gate']['relation']['s']
            config_data["bigger_size"]  = config_data["moe_intermediate_size"]//ss//config['partition']['n']
        config_data["carved"] = 1
        #改回去，因为按大expert计，具体激活情况用的acti_pattern,这个topk要和acti_pattern的长度一致
        config_data['num_experts_per_tok'] //= config['partition']['n'] 
        #用big的时候，不需要scale，因为不需要它来补偿softmax
        config_data['routed_scaling_factor'] //= config['partition']['n'] 


    if config['stage']==2:
        config_data["use_mapping"] = True
        config_data["expert_map"] = mapping
        config_data["dropped_num"] = config['drop']['list'][-1]
        if config['io']['model_type'] == 'olmoe':
            config_data['num_experts'] -= config['drop']['list'][-1]
        elif config['io']['model_type'] == 'deepseek':
            config_data['n_routed_experts'] -= config['drop']['list'][-1]
        elif config['io']['model_type'] == 'mixtral': 
            config_data['num_local_experts'] -= config['drop']['list'][-1]
        elif config['io']['model_type'] == 'qwen':
            config_data['num_experts'] -= config['drop']['list'][-1]

    
    config_to_path = os.path.join(save_dir, "config.json")
    with open(config_to_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    # print(f"Saved model and modified config at {save_dir}")
    return dict_to_namespace(config_data)

def save_full(model,config,yaml_path,mapping=None):
    copy_model_files_without_weights(config['io']['model_from'],config['io']['model_to'])
    model.save_pretrained(config['io']['model_to'], max_shard_size="8GB")
    cfg=save_model_config(config,yaml_path,mapping=mapping)