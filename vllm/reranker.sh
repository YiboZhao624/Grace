# /bin/bash

export CUDA_VISIBLE_DEVICES=0

vllm serve /root/path/to/BAAI-bge-reranker-v2-m3/\
    --port 8002\
    --gpu_memory_utilization 0.2\
