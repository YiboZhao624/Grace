python inference.py\
    --llm_name deepseek-chat \
    --url https://api.deepseek.com \
    --path \
    data/test_processed/3-shot-sys/QASPER-test-oracle-512cs-None-None-top3-wogt04.parquet \
    data/test_processed/3-shot-sys/QASPER-test-retriever_reranker-512cs-vllm_Qwen3-0.6B-vllm_-top3-wogt04.parquet \
    data/test_processed/3-shot-sys/QASPER-test-retriever-512cs-bm25-None-top3-wogt04.parquet \
    data/test_processed/3-shot-sys/QASPER-test-retriever-512cs-random-None-top3-wogt04.parquet \
    data/test_processed/3-shot-sys/QASPER-test-retriever-512cs-vllm_Qwen3-0.6B-None-top3-wogt04.parquet
    # data/test_processed/QASPER-test-retriever_reranker-512cs-vllm_Qwen3-0.6B-vllm_-top3-wogt04.parquet \
    # data/test_processed/QASPER-test-retriever-512cs-bm25-None-top3-wogt04.parquet \
    # data/test_processed/QASPER-test-retriever-512cs-random-None-top3-wogt04.parquet \
    # 

    