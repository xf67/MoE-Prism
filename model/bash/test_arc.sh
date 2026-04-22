MODEL=${1}
lm_eval --model vllm --model_args pretrained=$MODEL,dtype=bfloat16,gpu_memory_utilization=0.9 --tasks arc_easy,arc_challenge --num_fewshot 5 --batch_size 64 --trust_remote_code
