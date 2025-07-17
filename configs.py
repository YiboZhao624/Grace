from dataclasses import dataclass
from typing import List

@dataclass
class PreprocessConfig:
    dataset: list = ["qasper"]
    data_folder: List[str] = ["data/qasper"]


# each type of retriever has its corresponding parameters.
# here we don't distinguish them.
@dataclass
class RetrieverConfig:
    retriever_type: str = "bm25"
    # BM25 parameters
    k1: float = 1.5
    b: float = 0.75
    # sentence-transformer parameters
    model_name: str = "/root/shared_planing/LLM_model/Qwen3-Embedding-4B"
    device: str = "cuda:7"
    # vLLM parameters
    url: str = "http://localhost:8000"

# same for chunker.
@dataclass
class ChunkerConfig:
    chunking_strategy: str = "token"
    # TokenChunker parameters:
    model_name: str = "/root/shared_planing/LLM_model/Qwen3-Embedding-4B"
    max_length: int = 512
    overlap: int = 64

@dataclass
class DataGeneratorConfig:
    dataset: list = ["qasper"]
    paper_path: str = "./data/qasper/paper.json"
    qa_path: str = "./data/qasper/qa.json"
    top_k: int = 5
    wo_gt_evidence_rate: float = 0.2
    prompt_template: str = "default"
    number_template: bool = False
    recall_threshold: float = 0.5
    precision_threshold: float = 0.5


@dataclass
class Configs:
    preprocess_config: PreprocessConfig
    retriever_config: RetrieverConfig
    chunker_config: ChunkerConfig
    data_generator_config: DataGeneratorConfig