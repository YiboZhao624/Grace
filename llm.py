'''
This file contains the LLM class and its subclasses.
It is designed to be a wrapper of the LLM API.
Therefore anyone can define their own LLM class by inheriting from this class.
It will only used for testing the performance.
It won't be used for training.
We officially support the vllm as the basement.
'''
from configs import LLMConfig
import requests
import os
import openai

class LLM:
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def generate(self, prompt: str) -> str:
        raise NotImplementedError("You can't directly use the meta class. Subclasses must implement this method")

class vLLM(LLM):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.url
        self.model_name = config.model_name

    def generate(self, prompt:str) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = requests.post(url, headers=headers, json=data)
        print(response.json())
        return response.json()["choices"][0]["message"]["content"]

class GPT(LLM):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.url
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.model_name = config.model_name

    def generate(self, prompt:str) -> str:
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=512
        )
        return response.choices[0].message.content