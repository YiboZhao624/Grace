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
    model_name = "all-MiniLM-L6-v2",
    max_length = 512,
    overlap = 0
)

custom_DataGeneratorConfig = DataGeneratorConfig(
    dataset = ["qasper"],
    paper_path = "data/qasper/paper.json",
    qa_path = "data/qasper/qa.json",
    top_k = 5,
    wo_gt_evidence_rate = 0.2,
    prompt_template = "default",
    number_template = False
)

def main():
    preprocess_QASPER(custom_PreprocessConfig)
    retriever = get_retriever(custom_bm25_RetrieverConfig)
    chunker = get_chunker(custom_ChunkerConfig)
    data_generator = QASPERRetrieverDataGenerator(chunker, retriever, custom_DataGeneratorConfig)
    BM_25_data = data_generator.generate()

if __name__ == "__main__":
    main()