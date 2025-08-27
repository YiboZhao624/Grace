from dataclasses import dataclass, field
from typing import List, Literal, Union, TypeAlias
from itertools import product

@dataclass
class PreprocessConfig:
    dataset: list = field(default_factory=lambda: ["qasper"])
    data_folder: List[str] = field(default_factory=lambda: ["data/qasper"])


# each type of retriever has its corresponding parameters.
# here we don't distinguish them.
@dataclass
class RetrieverConfig:
    retriever_type: str = "bm25" # or sentence-transformer, or vLLM, or random
    # BM25 parameters
    k1: float = 1.5
    b: float = 0.75
    # sentence-transformer parameters
    model_name: str = "/root/shared_planing/LLM_model/Qwen3-Embedding-0.6B"
    device: str = "cuda:7"
    # vLLM parameters, model_name is shared with sentence-transformer.
    base_url: str = "http://localhost:8000"


# same for chunker.
@dataclass
class ChunkerConfig:
    chunking_strategy: Literal["greedy_sentence", "greedy_paragraph", "token", "sentence", "paragraph"] = "token"
    # TokenChunker parameters:
    model_name: str = "/root/shared_planing/LLM_model/Qwen3-Embedding-4B"
    max_length: int = 512
    overlap: int = 64

@dataclass
class RerankerConfig:
    # the configs for initializing the reranker. We only support vLLM for now.
    # To support transformers-based reranker, please modify the reranker.py file.
    reranker_type: str = "vllm"
    model_name: str = "/root/shared_planing/LLM_model/BAAI-bge-reranker-v2-m3/"
    device: str = "cuda:7"
    base_url: str = "http://localhost:8000"

Mode: TypeAlias = Literal["retriever", "retriever_reranker", "oracle", "random"]
ModeOrModes = Union[Mode, List[Mode]]
Split: TypeAlias = Literal["train", "val", "test"]
SplitOrSplits = Union[Split, List[Split]]

@dataclass
class DataGeneratorConfigs:
    # the configs for generating dataset, including train/dev/test split.
    dataset: str = "qasper"
    data_folder: str = "./data/qasper"
    split: SplitOrSplits = field(default_factory=lambda: ["train", "val", "test"])
    method: ModeOrModes = field(default_factory=lambda: ["retriever", "random", "oracle"])
    retriever_config: List[RetrieverConfig] = field(default_factory=lambda: [RetrieverConfig()])
    reranker_config: List[RerankerConfig] = field(default_factory=lambda: [RerankerConfig()])
    chunker_config: List[ChunkerConfig] = field(default_factory=lambda: [ChunkerConfig()])
    top_k: int = 5
    wo_gt_evidence_rate: float = 0.2
    prompt_template: str = "default"
    number_template: bool = False
    recall_threshold: float = 0.5
    precision_threshold: float = 0.5

    def iter_combinations(self):
        """
        To generate the combinations of the dataset, split and method.
        Allow the user generate a set of dataset with a single execution.
        return the dataset name, dataset folder, split and method in order.
        """
        if isinstance(self.split, str):
            self.split = [self.split]
        if isinstance(self.method, str):
            self.method = [self.method]
        for split, method in product(self.split, self.method):
            yield split, method

    @classmethod
    def from_dict(cls, cfg: dict):
        chunker_configs = [ChunkerConfig(**c) for c in cfg.get('chunker', [])]
        retriever_configs = [RetrieverConfig(**r) for r in cfg.get('retriever', [])]
        reranker_configs = [RerankerConfig(**rr) for rr in cfg.get('reranker', [])]

        return cls(
            dataset=cfg['data_generator']['dataset'],
            data_folder=cfg['data_generator']['data_folder'],
            split=cfg['data_generator']['split'],
            method=cfg['data_generator']['method'],
            top_k=cfg['data_generator']['top_k'],
            wo_gt_evidence_rate=cfg['data_generator']['wo_gt_evidence_rate'],
            prompt_template=cfg['data_generator']['prompt_template'],
            number_template=cfg['data_generator']['number_template'],
            recall_threshold=cfg['data_generator']['recall_threshold'],
            precision_threshold=cfg['data_generator']['precision_threshold'],
            chunker_config=chunker_configs,
            retriever_config=retriever_configs,
            reranker_config=reranker_configs,
        )

@dataclass
class DataGeneratorConfig:
    dataset: str = "qasper"
    data_folder: str = "./data/qasper"
    split: Split = "train"
    method: Mode = "retriever"
    retriever_config: RetrieverConfig = field(default_factory=lambda: RetrieverConfig())
    reranker_config: RerankerConfig = field(default_factory=lambda: RerankerConfig())
    chunker_config: ChunkerConfig = field(default_factory=lambda: ChunkerConfig())
    top_k: int = 5
    wo_gt_evidence_rate: float = 0.2
    prompt_template: str = "default"
    number_template: bool = False
    recall_threshold: float = 0.5
    precision_threshold: float = 0.5


@dataclass
class LLMConfig:
    # compatible with vLLM and GPT, for GPT, the url is not used as we deployed the openai package.
    model_name: str = "/root/shared_planing/LLM_model/Qwen3-8B"
    url: str = "http://localhost:8003"


@dataclass
class InferenceConfigs:
    # For initializing the LLM.
    llm_config: LLMConfig = field(default_factory=lambda: LLMConfig())
    # Dataset for inference.
    data_path: List[str] = field(default_factory=lambda: ["./data/processed/QASPER-3-methods-1000-samples-0805.parquet"])
    # see prompts.py for more details.
    prompt_template: str = "default"
    number_template: bool = False
    # For evaluation.
    metrics: List[str] = field(default_factory=lambda: ["RL", "BL", "EM", "BS", "LJ", "RR"])
    # For LJ, we need to provide the LLM API.
    LJ_api: LLMConfig = field(default_factory=lambda: LLMConfig())
    # Due to we have already built the dataset with retrieved evidence, we don't need to retrieve evidence again.
    # therefore the retriever and reranker are not involved.