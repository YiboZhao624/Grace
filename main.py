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
from configs import Configs, PreprocessConfig, RetrieverConfig, ChunkerConfig, DataGeneratorConfig
from data_generation import QASPERRetrieverDataGenerator
from utils import save_parquet

custom_PreprocessConfig = PreprocessConfig(
    dataset=["qasper"],
    data_folder=["data/qasper"]
)

custom_bm25_RetrieverConfig = RetrieverConfig(
    retriever_type="bm25",
    k1=1.5,
    b=0.75,
)

custom_ChunkerConfig = ChunkerConfig(
    chunking_strategy = "token",
    model_name = "/root/shared_planing/LLM_model/Qwen3-Embedding-4B",
    max_length = 512,
    overlap = 64
)

custom_DataGeneratorConfig = DataGeneratorConfig(
    dataset = ["qasper"],
    data_folder = "./data/qasper",
    split = "train",
    top_k = 5,
    wo_gt_evidence_rate = 0.2,
    prompt_template = "default",
    number_template = False
)

def generate_one_config(custom_DataGeneratorConfig: DataGeneratorConfig):
    preprocess_QASPER(custom_PreprocessConfig.data_folder[0])
    retriever = get_retriever(custom_bm25_RetrieverConfig)
    chunker = get_chunker(custom_ChunkerConfig)
    data_generator = QASPERRetrieverDataGenerator(chunker, retriever, custom_DataGeneratorConfig)
    BM_25_data = data_generator.generate()
    print(BM_25_data[0])
    file_name = f"./data/processed/QASPER_Split-{custom_DataGeneratorConfig.split}_Prompt-{custom_DataGeneratorConfig.prompt_template}_NumberTemplate-{custom_DataGeneratorConfig.number_template}_Retriever-{custom_bm25_RetrieverConfig.retriever_type}_TopK-{custom_DataGeneratorConfig.top_k}_WO_GT_Evidence_Rate-{custom_DataGeneratorConfig.wo_gt_evidence_rate}-test.parquet"
    save_parquet(BM_25_data, file_name)

if __name__ == "__main__":
    custom_data_generator_configs = []
    for custom_data_generator_config in custom_data_generator_configs:
        generate_one_config(custom_data_generator_config)