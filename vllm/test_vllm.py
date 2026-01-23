from warnings import resetwarnings
import requests
from typing import Union, List
import numpy as np
import urllib3

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def vllm_retrieve(base_url: str, model: str, documents: Union[list[str], str]) -> dict:
    if isinstance(documents, str):
        documents = [documents]
    url = f"{base_url}/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "input": documents,
    }
    response = requests.post(url, headers=headers, json=data, verify=False)
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
    response = requests.post(url, headers=headers, json=data, verify=False)
    print(response.json())
    return zip([response.json()["results"][i]["document"] for i in range(len(documents))], [response.json()["results"][i]["relevance_score"] for i in range(len(documents))])

def vllm_llm(base_url: str, model: str, prompt: str):
    url = f"{base_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(url, headers=headers, json=data, verify=False)
    print(response.json())
    return response.json()["choices"][0]["message"]["content"]

if __name__ == "__main__":
    base_url = f"http://localhost:8001"
    model  = "/root/path/to/Qwen3-Embedding-0.6B"
    documents = [
        "Reranking is fun!",
        "The capital of France is Paris",
        "vLLM is an open-source framework for fast AI serving",
    ]
    query = "What is the capital of France?"
    res = vllm_retrieve(base_url, model, documents)
    print(res)
    base_url = f"http://localhost:8002"
    model = "/root/path/to/BAAI-bge-reranker-v2-m3/"
    res = vllm_rerank(base_url, model, query, documents)
    print(res)
    base_url = f"http://localhost:8003" 
    model = "/root/path/to/Qwen2.5-7B-Instruct"
    prompt = "What is the capital of France?"
    res = vllm_llm(base_url, model, prompt)
    print(res)
