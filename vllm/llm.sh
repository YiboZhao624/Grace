# /bin/bash

export CUDA_VISIBLE_DEVICES=0,1

vllm serve /root/shared_planing/LLM_model/Qwen3-8B --port 8003 --tensor-parallel-size 2 --gpu-memory-utilization 0.7