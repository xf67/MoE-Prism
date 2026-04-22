torchrun --nproc-per-node 8 './src/train.py' -s 0 -l 1e-5 -e 1 -f uni_05_pa --saving_steps 1000 -m "./olmoeC/lin" --dataset pajama --sample 1 -li x -dy -k1 8 -k2 32 --mix_arc 0.01  --bf16 -bs 8

