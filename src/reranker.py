"""
This file is used to define the reranker class.
To faciliate the efficient reranking, we won't use the LLM as reranker.
We only use the specific retriever model to rerank the evidence.
The methods including: vllm reranker, transformers reranker.
"""

from transformers import AutoTokenizer
from typing import List, Dict, Any, Tuple, Optional
from configs import RerankerConfig
import requests
from utils import setup_logging

logger = setup_logging("Reranker")

class Reranker:
    '''
    This class is a meta-class for reranker.
    The mainly method is `rerank`.
    '''
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.reranker_client = None

    def _initialize_model(self):
        raise NotImplementedError("Subclasses must implement this method - _initialize_model")

    def rerank(self, evidence: List[str], query: str) -> List[str]:
        '''
        This method is used to rerank the evidence.
        Args:
            evidence: List[str], the evidence to be reranked.
            query: str, the query to be used for reranking.
        Returns:
            List[str], the reranked evidence.
        '''
        raise NotImplementedError("Subclasses must implement this method - rerank")


class vLLMReranker(Reranker):
    '''
    This class is used to rerank the evidences with vLLM.
    Please make sure the vLLM has been initialized.
    You should make sure the vllm serve model_name is the same with the input model_name.
    '''
    def __init__(self, config: RerankerConfig):
        super().__init__(config)
        self.base_url = f"{self.config.base_url}/v1/rerank"
        self.model_name = self.config.model_name
        logger.info(f"using the vllm reranker {self.model_name} at {self.base_url}")

    def rerank(self, evidence: List[str], query: str) -> List[str]:
        '''
        This method is used to rerank the evidence.
        Args:
            evidence: List[str], the evidence to be reranked.
            query: str, the query to be used for reranking.
        Returns:
            List[str], the reranked evidence.
        '''
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-fake-key"
        }
        
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": evidence,
            "top_n": len(evidence)
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error: Network request failed: {e}")
            logger.error(f"Payload: {payload}")
            raise
        result = response.json()
        ordered_evidence = [result["results"][i]["document"] for i in range(len(result["results"]))]
        reranked_index = [result["results"][i]["index"] for i in range(len(result["results"]))]
        return reranked_index, ordered_evidence


class TransformersReranker(Reranker):
    '''
    This class is used to rerank the evidences with transformers.
    Please make sure the transformers model has been initialized.
    '''
    def __init__(self, config: RerankerConfig):
        super().__init__(config)
        raise NotImplementedError("Subclasses must implement this method - TransformersReranker")


def get_reranker(reranker_config: RerankerConfig):
    '''
    get the reranker according to the configs.
    '''
    if reranker_config.reranker_type == "vllm":
        return vLLMReranker(reranker_config)
    elif reranker_config.reranker_type == "transformers":
        return TransformersReranker(reranker_config)
    else:
        raise ValueError(f"Unsupported reranker type: {reranker_config.reranker_type}")