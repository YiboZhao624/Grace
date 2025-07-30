# This file is used to define the retriever class.
# If you want to modify the retriever, just inherit the Retriever class and override the methods.

from typing import List, Dict, Optional, Union, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import vllm
from faiss import IndexFlatL2
from configs import RetrieverConfig
import requests

class Retriever:
    def __init__(self):
        self.model = self.load_model()

    def index(self, data: List):
        raise NotImplementedError("Subclasses must implement this method")

    def retrieve(self, query: str, top_k: int) -> Tuple[List[str], List[Tuple[int, float]]]:
        raise NotImplementedError("Subclasses must implement this method")

    def load_model(self):
        raise NotImplementedError("Subclasses must implement this method")

    def reset(self):
        raise NotImplementedError("Subclasses must implement this method")


class BM25Retriever(Retriever):
    def __init__(self, config: RetrieverConfig):
        # 调用父类构造函数
        super().__init__()
        # 初始化文档和倒排索引
        self.documents = []
        self.inverted_index = {}
        self.doc_freq = {}
        self.avgdl = 0
        self.k1 = config.k1
        self.b = config.b

    def load_model(self):
        # BM25不需要加载模型，这里返回None
        return None

    def reset(self):
        self.documents = []
        self.inverted_index = {}
        self.doc_freq = {}
        self.avgdl = 0

    def index(self, data: List[str]):
        # data: List of documents (str)
        self.documents = data
        self.inverted_index = {}
        self.doc_freq = {}
        doc_lens = []
        for idx, doc in enumerate(data):
            terms = doc.split()
            doc_lens.append(len(terms))
            seen = set()
            for term in terms:
                if term not in self.inverted_index:
                    self.inverted_index[term] = set()
                self.inverted_index[term].add(idx)
                if term not in seen:
                    self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
                    seen.add(term)
        self.avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 0
        self.doc_lens = doc_lens

    def retrieve(self, query: str, top_k: int) -> Tuple[List[str], List[Tuple[int, float]]]:
        # 计算BM25分数，返回top_k个文档
        import math
        query_terms = query.split()
        scores = [0.0 for _ in range(len(self.documents))]
        N = len(self.documents)
        for term in query_terms:
            if term not in self.inverted_index:
                continue
            df = self.doc_freq.get(term, 0)
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            for doc_id in self.inverted_index[term]:
                freq = self.documents[doc_id].split().count(term)
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_lens[doc_id] / self.avgdl)
                score = idf * freq * (self.k1 + 1) / denom if denom != 0 else 0
                scores[doc_id] += score
        # 获取分数最高的top_k个文档
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in ranked[:top_k]]
        return [self.documents[idx] for idx in top_indices], ranked

class SentenceTransformerRetriever(Retriever):
    def __init__(self, config: RetrieverConfig):
        super().__init__()
        self.config = config
        self.model = self.load_model()
        self.embeddings = []
        self.data = []
        self.faiss_index = None
        self.model_name = self.config.model_name
        self.device = self.config.device

    def reset(self):
        self.embeddings = []
        self.data = []
        self.faiss_index = None

    def load_model(self):
        return SentenceTransformer(self.model_name, device=self.device)

    def index(self, data: List[str]):
        self.embeddings = self.model.encode(data)
        self.data = data
        self.faiss_index = IndexFlatL2(self.embeddings.shape[1])
        self.faiss_index.add(self.embeddings)

    def retrieve(self, query: str, top_k: int) -> Tuple[List[str], List[Tuple[int, float]]]:
        query_embedding = self.model.encode(query)
        scores = self.model.similarity(query_embedding, self.embeddings)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.data[idx] for idx in top_indices], [(idx, score) for idx, score in enumerate(scores)]

class vLLMRetriever(Retriever):
    # you should launch the vLLM server first.
    def __init__(self, config:RetrieverConfig):
        super().__init__()
        self.config = config
        self.base_url = f"{self.config.base_url}/v1/embeddings"
        self.model_name = self.config.model_name
        self.embeddings = []
        self.data = []
        self.faiss_index = None
        print(f"using the vllm retriever {self.model_name} at {self.base_url}")

    def load_model(self):
        # there is no need to load model for vllm retriever.
        return None
    
    def reset(self):
        self.embeddings = []
        self.data = []
        self.faiss_index = None
    
    def call_model(self, input_text:Union[str, List[str]]) -> List[np.ndarray]:
        if isinstance(input_text, str):
            input_text = [input_text]
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "input": input_text,
        }

        response = requests.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return np.array([result["data"][i]["embedding"] for i in range(len(result["data"]))])
        # if isinstance(input_text, str):
        #     response = self.client.embed(input_text)
        #     return [response.embeddings[0]]
        # elif isinstance(input_text, List):
        #     response = self.client.embed(input_text)
        #     return response.embeddings

    def index(self, data: List[str]):
        self.embeddings = self.call_model(data)
        self.data = data
        self.faiss_index = IndexFlatL2(self.embeddings.shape[1])
        self.faiss_index.add(self.embeddings)
    
    def retrieve(self, query: str, top_k: int) -> Tuple[List[str], List[Tuple[int, float]]]:
        query_embedding = self.call_model(query)
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        return [self.data[idx] for idx in indices[0]], [(idx, score) for idx, score in enumerate(distances[0])]


def get_retriever(config: RetrieverConfig = None) -> Optional[Retriever]:
    """
    get retriever instance according to the distractor strategy
    
    Args:
        distractor_strategy: distractor strategy name
        config: retriever config
    
    Returns:
        Retriever instance or None
    """
    # default to use BM25Retriever, can be configured to other retriever
    retriever_type = config.retriever_type
    if retriever_type == "bm25":
        return BM25Retriever(config)
    elif retriever_type == "sentence_transformer":
        return SentenceTransformerRetriever(config)
    elif retriever_type == "vllm":
        return vLLMRetriever(config)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

if __name__ == "__main__":
    config = RetrieverConfig(
        retriever_type="vllm",
        model_name="/root/shared_planing/LLM_model/Qwen3-Embedding-0.6B",
        base_url = "http://localhost:8001"
    )
    retriever = get_retriever(config)
    retriever.index(["The capital of France is Paris", "Reranking is fun!", "vLLM is an open-source framework for fast AI serving"])
    print(retriever.retrieve("What is the capital of France?", 1))