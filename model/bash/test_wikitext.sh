MODEL=${1}
PEFT=${2}
COMMAND="pretrained=$MODEL"
if [ -n "$PEFT" ]; then
    COMMAND+=",enable_lora=True,lora_local_path=$PEFT"
fi
lm_eval --model vllm --model_args ${COMMAND},dtype=bfloat16,gpu_memory_utilization=0.9,max_length=2048 --tasks wikitext --num_fewshot 0 --batch_size 64 --trust_remote_code
