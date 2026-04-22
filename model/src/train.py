import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from datasets import load_dataset, concatenate_datasets, interleave_datasets
import shutil
import os
import glob
import json
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from AlternativeTrainer import AlternatingTrainer
SAVING_DIR_PREFIX = "./moe-gate-finetune"

# from accelerate import Accelerator
# accelerator = Accelerator()

import argparse
parser = argparse.ArgumentParser(description="ok")
parser.add_argument("-s","--noise", type= float, default=1e-5) 
parser.add_argument("-l","--lr", type= float, default=1e-5) 
parser.add_argument("-e","--epoch", type= int, default=1) 
parser.add_argument("-bs","--batch_size", type= int, default=2) # 4 for olmoe+fp32, 1 for deepseek+bf16
parser.add_argument("-f","--save_folder", type= str, help='Only folder name needed')  
parser.add_argument("-m","--model_path", type= str)  
parser.add_argument("-n","--model_name", type= str)  
parser.add_argument("--saving_steps", type= int, default=3000) 
parser.add_argument("--dataset", type= str, default='wikitext')  
parser.add_argument("--sample", type= float, default=1.0)   # 数据集上随机采样
parser.add_argument("--split", type= str, default='train')  # 有的数据集里叫train_sft
parser.add_argument("--mix_arc", type= float, default=0.0) #数值表示ARC在整体里的占比

parser.add_argument("-k","--target_k", type= int, default=16, help="Target top_k value for experts.") # naive训练时可以自定k，在保存时会覆盖在模型的config里

parser.add_argument("-dy","--use_dynamic_k",  action="store_true") # 训练时用动态k
parser.add_argument("-k1","--initial_k", type= int, default=16, help="Initial top_k value for experts.")
parser.add_argument("-k2","--final_k", type= int, default=32, help="Final top_k value for experts.")

parser.add_argument("-ma","--use_mask",  action="store_true") # 训练一个mask
parser.add_argument("--alternative", action="store_true") 

parser.add_argument("-li","--fix_list",  type= str) # 用固定list，传list编号进来（如4431）,或者传个x，用dynamicK

parser.add_argument("--bf16", action="store_true") 
parser.add_argument("--block_size", type= int, default=2048) 

parser.add_argument("--local_rank", type=int, default=-1)  #兼容deepspeed传入


args = parser.parse_args()

assert(not (args.use_dynamic_k and args.use_mask))


uni_list32 = [0,1,2,4, 3,5,6,8, 7,9,10,12, 11,13,14,16, 15,17,18,20, 19,21,22,24, 23,25,26,28, 27,29,30,31]
uni_list24 = [0,1,2,4, 3,5,6,8, 7,9,10,12, 11,13,14,16, 15,17,18,20, 19,21,22,23]
uni_list16 = [0,1,2,4, 3,5,6,8, 7,9,10,12, 11,13,14,15]
uni_list8 = [0,1,2,4, 3,5,6,7]

# --------------------------------------------------
# 1. 配置模型和数据集
# --------------------------------------------------
model_id = args.model_path # "./olmoeC/lin"
assert (isinstance(model_id,str))
if not args.model_name:
    if 'olmoe' in model_id.lower():
        model_name = 'olmoe' 
        uni_list=uni_list32
    elif 'deepseek' in model_id.lower() or 'dpsk' in model_id.lower() or 'moon' in model_id.lower():
        model_name = 'deepseek'
        uni_list=uni_list24
    elif 'qwen' in model_id.lower():
        model_name = 'qwen'
        uni_list=uni_list16
    elif 'mixtral' in model_id.lower():
        model_name = 'mixtral'
        uni_list=uni_list8
    else:
        print("Error: Unkown model_name")
        exit()
else:
    model_name = args.model_name

if args.dataset=='wikitext':
    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1"
elif args.dataset=='pajama':
    dataset_name = "DKYoon/slimpajama-200k"
    dataset_config = None
else:
    print("Error: Not supported dataset")
    raise(NotImplementedError)

sample_ratio = args.sample
noise_std = args.noise
num_cores = 16

predefined_list={
                '431':[0,1,2,3,4,5,6,8],
                '4431':[0,1,2,3,4,5,6,7,8,9,10,12,],
                '443221':[0,1,2,3,4,5,6,7,8,9,10,12,13,16,17,20],
                '44431': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16],
                '4443221':[0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14, 16,17, 20,21, 24],
                '444431': [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18, 20,]
                }



# --------------------------------------------------
# 2. 加载模型和分词器
# --------------------------------------------------
print("--- Loading model and tokenizer ---")

# with deepspeed.zero.Init():
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    # device_map="auto",
    trust_remote_code=True,
)
max_length = model.config.max_position_embeddings


if args.fix_list:
    if args.fix_list in predefined_list: # 后面别用这个了，uni_list完全覆盖这个
        fix_list=predefined_list[args.fix_list]
        args.target_k=max(fix_list)+1
        for layer in model.model.layers:
            layer.mlp.use_list=True
            layer.mlp.use_mask=False
            layer.mlp.selection_mask=torch.tensor(fix_list)
            layer.mlp.max_topk = max(fix_list)+1
    else:
        print("Use uni_list")
        fix_list=uni_list
        args.target_k=max(fix_list)+1
        for layer in model.model.layers:
            layer.mlp.use_list=True
            layer.mlp.use_mask=False
            layer.mlp.selection_mask=torch.tensor(fix_list)
            layer.mlp.max_topk = max(fix_list)+1
else:
    fix_list=None

if args.use_mask:
    args.target_k = model.model.layers[1].mlp.gate_mask.shape[0] # 覆盖target_k到最大
# 不然就用传入的target_k

for layer in model.model.layers: # 这个写入之后，后面call_back调dynamicK的时候会接着盖掉
    layer.mlp.top_k = args.target_k

tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
model.config.use_cache=False
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = 'right' # Set padding side to 'right' for Causal LM
if tokenizer.chat_template is None:
    print("Tokenizer does not have a chat template. Setting a default one.")
    tokenizer.chat_template = "{{ eos_token }}{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}" # from olmoe-7B-Instruct


# --------------------------------------------------
# 3. 核心步骤：冻结除门控层外的所有参数
# --------------------------------------------------
print("--- Freezing model parameters ---")

for param in model.parameters():
    param.requires_grad = False

# 只解冻门控层的权重
trainable_layers = ["mlp.gate"]

for name, param in model.named_parameters():
    if any(trainable_layer in name for trainable_layer in trainable_layers):
        if 'moon' in model_id.lower() and "e_score_correction_bias" in name:
            continue
        param.requires_grad = True
        print(f"Unfrozen: {name}")

print("--- Re-initializing gate weights ---")
if noise_std>0:
    for name, param in model.named_parameters():
        if param.requires_grad and "weight" in name:
            # nn.init.normal_(param.data, mean=0.0, std=0.02)
            noise = torch.randn_like(param.data) * noise_std
            param.data.add_(noise)
            print(f"Re-initialized with normal distribution: {name}")

# --------------------------------------------------
# 4. 验证参数冻结情况
# --------------------------------------------------
def print_trainable_parameters(model):
    """
    打印模型中可训练参数的数量。
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )

print("--- Parameter status ---")
print_trainable_parameters(model)

# --------------------------------------------------
# 5. 加载和预处理数据集
# --------------------------------------------------
print("--- Loading and processing dataset ---")
dataset = load_dataset(dataset_name, dataset_config, split=args.split)

if sample_ratio < 1.0:
    total_samples = len(dataset)
    num_samples = int(total_samples * sample_ratio)
    dataset = dataset.shuffle(seed=42).select(range(num_samples))

# 数据集预处理函数
def tokenize_and_chunk(examples):
    # 将所有文本连接起来，然后切分成固定长度的块
    block_size = args.block_size
    
    # 将所有文本连接成一个长字符串
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # 我们丢弃最后不足一个block_size的部分
    total_length = (total_length // block_size) * block_size
    
    # 切分成块
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

if dataset_name=='wikitext' or dataset_name=='DKYoon/slimpajama-200k':

    namae = "text" if dataset_name=='wikitext' else "input" #pajama
    original_columns = dataset.column_names 

    # 对数据集进行分词
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples[namae]),
        batched=True,
        remove_columns=original_columns 
    )

    # 对分词后的数据进行分块
    chunked_dataset = tokenized_dataset.map(
        tokenize_and_chunk,
        batched=True,
    )
else:
    raise(NotImplementedError)


if args.mix_arc>0:
    assert(args.mix_arc<=1)
    print("\n--- Loading and processing ARC dataset for mixing ---")
    
    # 1. 加载ARC数据集
    arc_dataset = load_dataset("ai2_arc", 'ARC-Challenge', split="train")

    # 2. 预处理ARC数据集，将其转换为文本序列
    def format_arc_example(example):
        # 答案标签映射 (e.g., "1" -> "A", "B" -> "B")
        answer_key = example['answerKey']
        choices = example['choices']
        
        # 构建选项文本
        choice_texts = []
        for i, label in enumerate(choices['label']):
            text = choices['text'][i]
            choice_texts.append(f"{label}) {text}")
        
        # 将问题和选项组合成一个字符串
        prompt = f"Question: {example['question']}\n\n"
        prompt += "Choices:\n" + "\n".join(choice_texts)
        
        # 找到正确答案的文本
        try:
            # 处理数字和字母标签
            if answer_key.isdigit():
                correct_choice_index = int(answer_key) - 1
            else: # A, B, C, D
                correct_choice_index = choices['label'].index(answer_key)
            correct_answer_text = choices['text'][correct_choice_index]
            correct_answer_label = choices['label'][correct_choice_index]
            
            # 完整的文本序列，包含问题、选项和答案
            full_text = f"{prompt}\n\nAnswer: {correct_answer_label}) {correct_answer_text}"
        except (ValueError, IndexError):
            # 如果答案标签有问题，跳过这个样本
            return {"text": ""}

        return {"text": full_text}

    arc_dataset_formatted = arc_dataset.map(format_arc_example, remove_columns=arc_dataset.column_names)
    
    # 过滤掉处理失败的空样本
    arc_dataset_formatted = arc_dataset_formatted.filter(lambda example: len(example['text']) > 0)
    
    # 对ARC数据进行分词
    tokenized_arc_dataset = arc_dataset_formatted.map(
        lambda examples: tokenizer(examples['text']), # 注意这里是 'text'
        batched=True,
        remove_columns=['text']  # 移除原始文本列，留下 input_ids 和 attention_mask
    )

    chunked_arc_dataset = tokenized_arc_dataset.map(
        tokenize_and_chunk,
        batched=True,
    )

    print(f"Original general dataset size: {len(chunked_dataset)}")
    print(f"ARC dataset size: {len(chunked_arc_dataset)}")
    
    # 3. 合并数据集
    # 确保两个数据集有相同的列 (input_ids, attention_mask, labels)
    # chunked_dataset 已经有了这些列
    # 确保 tokenized_arc_dataset 也有
    datasets_to_interleave = [chunked_dataset, chunked_arc_dataset]
    
    if args.mix_arc==1:
        final_dataset = concatenate_datasets([chunked_dataset, chunked_arc_dataset])
    else:
        probabilities = [1-args.mix_arc, args.mix_arc]
        # 使用 interleave_datasets 进行混合
        # stopping_strategy='all_exhausted' 表示直到所有数据集的样本都被抽取完才停止
        # 这意味着较小的数据集会被重复使用，以满足混合比例
        final_dataset = interleave_datasets(
            datasets_to_interleave, 
            probabilities=probabilities, 
            seed=42, 
            stopping_strategy='all_exhausted'
        )
    print(f"Combined dataset size: {len(final_dataset)}")

    # 4. 打乱合并后的数据集
    final_dataset = final_dataset.shuffle(seed=42)
    print("--- Datasets mixed and shuffled successfully! ---")
else:
    final_dataset = chunked_dataset

# --------------------------------------------------
# 6. 配置训练参数
# --------------------------------------------------
print("--- Configuring training ---")

class TopKAnnealingCallback(TrainerCallback):
    """
    A callback to LINEARLY adjust the top_k parameter of MoE layers during training.
    """
    def __init__(self, model, initial_k, final_k):
        super().__init__()
        self.model = model
        self.initial_k = initial_k
        self.final_k = final_k
        self.current_k = None

    def on_step_begin(self, args, state, control, **kwargs):
        """
        Called at the beginning of each training step.
        Calculates the target k based on linear interpolation.
        """
        # Calculate training progress (from 0.0 to 1.0)
        progress = state.global_step / state.max_steps
        
        # Linear interpolation: k = initial_k + progress * (final_k - initial_k)
        k_range = self.final_k - self.initial_k
        target_k_float = self.initial_k + (k_range * progress)
        
        # Round to the nearest integer, as top_k must be an integer
        target_k = int(round(target_k_float))
        
        # If the target_k has changed, update the model and log it
        if target_k != self.current_k:
            print(f"Step {state.global_step}/{state.max_steps} (Progress: {progress:.2f}): "
                  f"Changing top_k from {self.current_k} to {target_k}")
            self.current_k = target_k
            for layer in self.model.model.layers:
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "top_k"):
                    layer.mlp.top_k = target_k

    def on_train_begin(self, args, state, control, **kwargs):
        """
        Set the initial top_k value at the very beginning of training.
        """
        print(f"Training started. Setting initial top_k to {self.initial_k}")
        self.current_k = self.initial_k
        for layer in self.model.model.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "top_k"):
                layer.mlp.top_k = self.initial_k

class MaskAnnealingCallback(TrainerCallback): 
    """
    A callback to LINEARLY adjust the top_k parameter of MoE layers during training.
    """
    def __init__(self, model,):
        super().__init__()
        self.model = model
        self.current_t = 1
        self.train_which = 0

    def on_step_begin(self, args, state, control, **kwargs):
        """
        Called at the beginning of each training step.
        Calculates the target t based on linear interpolation.
        """
        # Calculate training progress (from 0.0 to 1.0)
        progress = state.global_step / state.max_steps
        
        # Linear interpolation: 
        target_t = 1 - progress
        
        # If the target_k has changed, update the model and log it
        self.current_t = target_t
        for layer in self.model.model.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "tau"):
                layer.mlp.tau = target_t
            else:
                print("ERROR: layer.mlp.tau not found ")

    def on_train_begin(self, args, state, control, **kwargs):
        """
        Set the initial top_k value at the very beginning of training.
        """
        print(f"Training started. Setting initial tau to 1")
        self.current_k = 1
        for layer in self.model.model.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "tau"):
                layer.mlp.tau = 1


class CustomSavingCallback(TrainerCallback):
    """
    Save custom config
    """
    def __init__(self, source_dir,use_mask,max_topk,use_list,fixed_list,target_k):
        """
        Args:
            custom_code_source_dir (str): 自定义代码文件所在的源目录。
            file_names (list): 需要复制的文件名列表。
        """
        self.source_dir = source_dir
        search_path = os.path.join(self.source_dir, '*.py')
        all_py_files = [os.path.basename(p) for p in glob.glob(search_path)]
        self.file_names = all_py_files
        # print(self.file_names)
        self.use_mask = use_mask
        self.max_topk = max_topk
        self.use_list = use_list
        self.fixed_list = fixed_list
        self.target_k = target_k

    def on_save(self, args, state, control, **kwargs):
        global_step = state.global_step
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{global_step}"
        checkpoint_dir = os.path.join(args.output_dir, checkpoint_folder)
        print(f"--- Copying custom code to {checkpoint_dir} ---")
        
        for file_name in self.file_names:
            source_file = os.path.join(self.source_dir, file_name)
            destination_file = os.path.join(checkpoint_dir, file_name)
            try:
                shutil.copy2(source_file, destination_file) # copy2 会保留元数据
                # print(f"Successfully copied {file_name} to {destination_file}")
            except Exception as e:
                print(f"Error copying {file_name}: {e}")
        
        config_from_path = os.path.join(self.source_dir, "config.json")
        with open(config_from_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        config_data['use_mask'] = self.use_mask
        config_data['max_topk'] = self.max_topk 
        config_data['use_list'] = self.use_list 
        config_data['fixed_list'] = self.fixed_list
        if self.use_list:
            config_data['num_experts_per_tok'] = self.max_topk
        else:
            config_data['num_experts_per_tok'] = self.target_k

        config_to_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_to_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)


training_args = TrainingArguments(
    output_dir=f"{SAVING_DIR_PREFIX}-{model_name}/{args.save_folder}",
    per_device_train_batch_size=args.batch_size, 
    gradient_accumulation_steps=8, 
    learning_rate=args.lr,         
    num_train_epochs=args.epoch,    
    logging_steps=1,              # 每10步记录一次日志
    # eval_strategy="epoch",  
    # eval_steps=100,  
    # save_strategy="epoch", 
    save_steps = args.saving_steps,
    bf16=True if args.bf16 else False,       
    # optim="adamw_bnb_8bit" if args.bf16 else None,              
    gradient_checkpointing=False,   # 因为只有gate，所以不能开，不然没梯度
    report_to="none",              # "wandb" 或 "tensorboard"
    ddp_find_unused_parameters=True,
    # dataloader_num_workers=16,
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    # adam_epsilon=1e-6
)

if dataset_name == 'wikitext' or dataset_name=='DKYoon/slimpajama-200k':
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
else:
    raise(NotImplementedError)

top_k_callback = TopKAnnealingCallback(
    model=model,
    initial_k=args.initial_k,
    final_k=args.final_k
)


mask_callback = MaskAnnealingCallback(
    model=model,
)

save_callback = CustomSavingCallback(
    source_dir=model_id,
    use_mask=args.use_mask,
    max_topk=args.target_k, # 对于用list的情况，随便写进去就好了，带list的时候在初始化的时候会重置max_topk为max(list)+1
    use_list=True if args.fix_list else False,
    fixed_list=fix_list,
    target_k = args.target_k
)

# --------------------------------------------------
# 7. 初始化并开始训练
# --------------------------------------------------
call_backs=[save_callback]
if args.use_dynamic_k:
    call_backs.append(top_k_callback)
if args.use_mask:
    call_backs.append(mask_callback)
    # call_backs.append(top_k_callback)
if args.alternative and args.use_mask:
    print("Called alternatingTrainer")
    trainer = AlternatingTrainer(
        model=model,
        args=training_args,
        train_dataset=final_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=call_backs, 
    )
else:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=call_backs, 
    )

print("--- Starting training ---")


trainer.train()

