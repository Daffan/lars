from typing import Tuple, Union, List
from abc import ABC, abstractmethod

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, DebertaV2Model, AutoModel


class BaseEncoder(ABC):
    @abstractmethod
    def get_embedding_size(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def encode(self, input: str) -> Union[np.ndarray, torch.tensor]:
        raise NotImplementedError
    
    @abstractmethod
    def encode(self, input_batch: List[str]) -> Union[np.ndarray, torch.tensor]:
        raise NotImplementedError
    

class BERTEncoder(BaseEncoder):
    def __init__(self, model_config="bert-base-uncased", device=None) -> None:
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model_config = model_config
        self.tokenizer = AutoTokenizer.from_pretrained(model_config)
        print("model_config:", model_config)
        if model_config == "bert-base-uncased":
            self.model = AutoModelForTokenClassification.from_pretrained(model_config).to(self.device)
        elif model_config == "microsoft/deberta-v2-xlarge":
            self.model = DebertaV2Model.from_pretrained(model_config).to(self.device)
        elif model_config == "sentence-transformers/all-MiniLM-L6-v2":
            self.model = AutoModel.from_pretrained(model_config).to(self.device)

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

        if self.model_config == "bert-base-uncased":
            self.embedding_size = 768
        elif self.model_config == "microsoft/deberta-v2-xlarge":
            self.embedding_size = 1536
        elif self.model_config == "sentence-transformers/all-MiniLM-L6-v2":
            self.embedding_size = 384
        else:
            raise ValueError(f"The embedding for model: {self.model_config} is not decided!")

    def get_embedding_size(self) -> int:
        return self.embedding_size
        
    def encode_batch(self, input_batch: List[str], cpu=True, batch_size=20) -> torch.tensor:
        input_mini_batches = [input_batch[i:i + batch_size] for i in range(0, len(input_batch), batch_size)]
        sentence_embeddings = []
        for i, input_mini_batch in enumerate(input_mini_batches):
            print(f"Computing embeddings {i}/{len(input_mini_batches)}", end="\r")
            input = self.tokenizer(input_mini_batch, truncation=True, padding=True, return_tensors="pt").to(self.device)
            output = self.model(**input, output_hidden_states=True)
            last_hidden_states = output.hidden_states[-1]
            sentence_embedding = last_hidden_states[:, 0, :]
            if cpu:
                sentence_embeddings.append(sentence_embedding.detach().cpu().numpy())
            else:
                sentence_embeddings.append(sentence_embedding)
        if cpu:
            sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)
        else:
            sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
        return sentence_embeddings
    
    def encode(self, input: str, cpu=True) -> torch.tensor:
        return self.encode_batch([input], cpu)[0, :]
        

class DummyEncoder(BaseEncoder):
    def get_embedding_size(self) -> int:
        return 10
    
    def encode(self, input_batch: List[str]) -> np.ndarray:
        return np.zeros(10)
    

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

class OpenAIEncoder(BaseEncoder):
    def __init__(self, model="text-embedding-ada-002"):
        self.model = model

    def get_embedding_size(self) -> int:
        if 'code-search' in self.model:
            return 2048
        else:
            return 1536
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def encode_batch(self, input_batch: List[str], cpu=True, batch_size=20) -> torch.tensor:
        embeddings = []
        for i, input in enumerate(input_batch):
            print(f"Computing embeddings {i}/{len(input_batch)}", end="\r")
            response = openai.Embedding.create(
                model=self.model,
                input=input)
            embedding = np.array(response["data"][0]["embedding"])
            embeddings.append(embedding)
        return np.stack(embeddings, axis=0)
    
    def encode(self, input: str, cpu=True) -> torch.tensor:
        return self.encode_batch([input], cpu)[0, :]

if __name__ == "__main__":
    # encoder = BERTEncoder(model_config="microsoft/deberta-v2-xlarge")
    # encoder = BERTEncoder(model_config="sentence-transformers/all-MiniLM-L6-v2")
    encoder = OpenAIEncoder(model="code-search-babbage-text-001")
    aaa = encoder.encode("hello world")
    print(aaa)
    print(len(aaa), encoder.get_embedding_size())