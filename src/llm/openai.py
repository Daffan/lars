import os
import openai
import time

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from src.llm.base import BaseLLM

openai.api_key = os.getenv("OPENAI_API_KEY")

openai_default_parameters = {
    "temperature": 0.0,
    "max_tokens": 512,
    "top_p": 1.0,
    "stop": ["\n\n"]
}

class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model_name: str,
        temperature=0.01,
        top_p=0.6,
        max_new_tokens=256,
        stop=["\n\n"]):
        
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_new_tokens
        self.stop = stop

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def predict(self, prompt: str) -> str:
        if self.model_name in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k"]:
            messages = [
                {"role": "user", "content": prompt},
            ]
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stop=self.stop)
            output = response["choices"][0]["message"]["content"].strip()
        else:
            response = openai.Completion.create(
                engine=self.model_name,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stop=self.stop)
            output = response["choices"][0]["text"].strip()
        return output