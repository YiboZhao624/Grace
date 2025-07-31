# /bin/bash

export CUDA_VISIBLE_DEVICES=0

vllm serve /root/shared_planing/LLM_model/Qwen3-Embedding-0.6B\
    --port 8001\
    --gpu_memory_utilization 0.2\