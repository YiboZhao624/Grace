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
    retriever_type: str
    # BM25 parameters
    k1: float
    b: float
    # sentence-transformer parameters
    model_name: str
    device: str
    # vLLM parameters
    url: str

# same for chunker.
@dataclass
class ChunkerConfig:
    chunking_strategy: str
    # TokenChunker parameters:
    model_name: str
    max_length: int
    overlap: int

@dataclass
class DataGeneratorConfig:
    dataset: list = ["qasper"]
    paper_path: str
    qa_path: str
    top_k: int
    wo_gt_evidence_rate: float
    prompt_template: str
    number_template: bool


@dataclass
class Configs:
    preprocess_config: PreprocessConfig
    retriever_config: RetrieverConfig
    chunker_config: ChunkerConfig
    data_generator_config: DataGeneratorConfig