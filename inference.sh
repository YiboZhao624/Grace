python inference.py\
    --llm_name /root/projects/RAGRL/ckpts/DAPO-0926-reward-v2-full/0929-ckpt_550 \
    --url http://localhost:8004 \
    --path \
    data/processed/QASPER-test-oracle-512cs-None-None-top3-wogt04.parquet \
    data/processed/QASPER-test-retriever_reranker-512cs-vllm_Qwen3-0.6B-vllm_-top3-wogt04.parquet 

    # data/processed/QASPER-test-retriever-512cs-bm25-None-top3-wogt04.parquet \
    # data/processed/QASPER-test-retriever-512cs-random-None-top3-wogt04.parquet \
    # data/processed/QASPER-test-retriever-512cs-vllm_Qwen3-0.6B-None-top3-wogt04.parquet