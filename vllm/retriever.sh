# /bin/bash

export CUDA_VISIBLE_DEVICES=0

vllm serve /root/path/to/Qwen3-Embedding-0.6B\
    --port 8001\
    --gpu_memory_utilization 0.2\
