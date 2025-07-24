'''
This file contains the LLM class and its subclasses.
It is designed to be a wrapper of the LLM API.
Therefore anyone can define their own LLM class by inheriting from this class.
It will only used for testing the performance.
It won't be used for training.
We officially support the vllm as the basement.
'''
from configs import LLMConfig
import vllm

class LLM:
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def generate(self, prompt: str, max_tokens: int) -> str:
        raise NotImplementedError("You can't directly use the meta class. Subclasses must implement this method")

class vLLM(LLM):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        raise NotImplementedError("You can't directly use the meta class. Subclasses must implement this method")