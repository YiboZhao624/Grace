from warnings import resetwarnings
import requests
from typing import Union, List
import numpy as np

documents = [
    "Reranking is fun!",
    "The capital of France is Paris",
    "vLLM is an open-source framework for fast AI serving",
]

query = "What is the capital of France?"

model = "/root/shared_planing/LLM_model/Qwen3-Embedding-0.6B"

def vllm_retrieve(base_url: str, model: str, documents: Union[list[str], str]) -> dict:
    if isinstance(documents, str):
        documents = [documents]
    url = f"{base_url}/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "input": documents,
    }
    response = requests.post(url, headers=headers, json=data)
    return np.array([response.json()["data"][i]["embedding"] for i in range(len(documents))])

def vllm_rerank(base_url: str, model: str, query: str, documents: List[str]):
    url = f"{base_url}/v1/rerank"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": len(documents)
    }
    response = requests.post(url, headers=headers, json=data)
    print(response.json())
    return zip([response.json()["results"][i]["document"] for i in range(len(documents))], [response.json()["results"][i]["relevance_score"] for i in range(len(documents))])


if __name__ == "__main__":
    base_url = f"http://localhost:8002" 
    model = "/root/shared_planing/LLM_model/BAAI-bge-reranker-v2-m3/"
    res = vllm_rerank(base_url, model, query, documents)
    print([entry[1] for entry in res])