# MoE-Prism


## Model Refactoring Engine

Usage: see bash scripts in model/bash

`SA` is for model refactoring and generating training-free gate.

`Lin` is for model refactoring with linear gate which is used for finetuning.

`FT` is for lightweight gat finetuning.

And you can use `lm_eval` to test model accuracy with `test_arc.sh` and `test_wikitext.sh`.


## Top-k-aware Runtime Scheduler & Executor

### Install

```
cd vllm
VLLM_USE_PRECOMPILED=1 pip3 install --editable .
pip3 install debugpy
pip3 install -U huggingface_hub
mkdir test
```

The `test` folder will be used to save some logs and experiment results.


### Get performance model

```bash
cd vllm

# run it in one bash window
QOS_AWARE=2 QOS_K_LIST=r1,8 SCHED_MODE=fifo vllm serve <PATH_TO_YOUR_MODEL> --no-enable-prefix-caching --compilation-config '{"cudagraph_mode":"PIECEWISE","share_attn_cudagraph_across_topk":true}'

# run it in another bash window
bash './experiment/profile_prefill.sh'

```

`QOS_K_LIST=r1,8` means that the supported topk range is 1~8, you may change it to `r1,24` or `r1,32` etc.

After this, you will get something like 

### Start speed benchmark

```bash
cd vllm

# run it in one bash window
bash  './experiment/start_server.sh' fifo

# run it in another bash window
bash './experiment/run_random_bench.sh'

```

