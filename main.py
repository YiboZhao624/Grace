# this is the main file for the data generation.
# You can use this file for generating the data for the veRL training stage.
# the optional stage is the difficulty test stage.
# the main stages including:
# 0. parse the arguments.
# 1. preprocess the data.
# 2. initialize the chunker, retriever, and then a kind of data generator.
# 3. generate the data.
# 4. save the data.
# 5. loop the process 2 - 4 for generating more kinds of data.
# 6. test the difficulty of each kind of data.

from preprocess import preprocess_QASPER
from argparse import ArgumentParser
from retriever import get_retriever
from chunker import get_chunker
from configs import PreprocessConfig, RetrieverConfig, ChunkerConfig, DataGeneratorConfigs, RerankerConfig
from data_generation import QASPERDataGenerator, QASPERRetrieverDataGenerator, QASPEROracleDataGenerator, QASPERRetrieverRerankerDataGenerator, get_data_generator
from utils import save_parquet, setup_logging

logger = setup_logging("Data Generation")


custom_PreprocessConfig = PreprocessConfig(
    dataset=["qasper"],
    data_folder=["data/qasper"]
)

custom_RetrieverConfigs = [RetrieverConfig(retriever_type="bm25",k1=1.5,b=0.75,),
                                RetrieverConfig(retriever_type="vllm",base_url="http://localhost:8001",),
                                RetrieverConfig(retriever_type="vllm",base_url="http://localhost:8001",),
                            ]

custom_RerankerConfigs = [RerankerConfig(reranker_type="vllm",model_name="/root/shared_planing/LLM_model/BAAI-bge-reranker-v2-m3/",base_url="http://localhost:8002",),
                            RerankerConfig(reranker_type="vllm",model_name="/root/shared_planing/LLM_model/BAAI-bge-reranker-v2-m3/",base_url="http://localhost:8002",),
                            RerankerConfig(reranker_type="vllm",model_name="/root/shared_planing/LLM_model/BAAI-bge-reranker-v2-m3/",base_url="http://localhost:8002",),
                            ]

custom_ChunkerConfig = ChunkerConfig(
    chunking_strategy = "token",
    model_name = "/root/shared_planing/LLM_model/Qwen3-Embedding-0.6B",
    max_length = 256,
    overlap = 64
)

custom_DataGeneratorConfig = DataGeneratorConfigs(
    dataset = "qasper",
    data_folder = "./data/qasper",
    split = ["train", "val", "test"],
    method = ["oracle"],
    top_k = 1,
    wo_gt_evidence_rate = 0.2,
    prompt_template = "default",
    number_template = False,
    chunker_config = [custom_ChunkerConfig],
    retriever_config = None,
    reranker_config = None,
)


if __name__ == "__main__":
    custom_data_generator_configs = custom_DataGeneratorConfig
    for data_generator in get_data_generator(custom_data_generator_configs):
        data = data_generator.generate_all()
        for split_method, entries in data.items():
            split = split_method.split("_")[0]
            method = split_method.split("_")[1]
            file_name = f"./data/processed/0806-256cs/QASPER_Split-{split}_Prompt-{custom_data_generator_configs.prompt_template}_NumberTemplate-{custom_data_generator_configs.number_template}_{method}.parquet"
            save_parquet(entries, file_name)

# file_name = f"./data/processed/0806-512cs/QASPER_Split-{split}_Prompt-{data_generator_config.prompt_template}_NumberTemplate-{data_generator_config.number_template}_{method}_{data_generator.retriever.retriever_type}_TopK-{data_generator_config.top_k}_WO_GT_Evidence_Rate-{data_generator_config.wo_gt_evidence_rate}.parquet"