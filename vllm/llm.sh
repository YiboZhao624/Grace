# /bin/bash

export NCCL_P2P_DISABLE=1 
export CUDA_VISIBLE_DEVICES=3,4
python3 -m vllm.entrypoints.openai.api_server\
    --model /root/shared_planing/LLM_model/Qwen2.5-7B-Instruct\
    --port 8003\
    --dtype bfloat16\
    --gpu-memory-utilization 0.9\
    --max-model-len 16000\
    --tensor-parallel-size 1