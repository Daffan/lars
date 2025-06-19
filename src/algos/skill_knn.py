## Implement Skill-KNN from `Skill-based few-shot selection for in-context learning`

import json
import os
import random

import numpy as np
import torch

from src.envs.cot_env import CoTDatasetEnv
from src.algos.base import SelectionMethod
from src.envs.encoders import BaseEncoder, BERTEncoder, OpenAIEncoder
from src.llm import endpoint_to_class

class SkillKNN(SelectionMethod):
    def __init__(
        self,
        env: CoTDatasetEnv,
        encoder_type: str,
        task: str,
        endpoint: str,
        rewrite_cand_path: str,  # need to make sure rewrite cand is the same split as the cand
        device: str = "cpu",
        metric: str = "cosine",
        seed: int = 11,
        temperature: float = 0.01,
        top_p: float = 0.6,
        max_new_tokens: int = 512,
        num_examples: int = 8
    ):
        super().__init__(env, seed)
        # seed random
        np.random.seed(seed)
        random.seed(seed)

        self.env = env
        self.encoder_type = encoder_type
        self.seed = seed
        self.metric = metric
        self.device = device
        
        if encoder_type in ["text-embedding-ada-002", "code-search-babbage-text-001"]:
            self.encoder = OpenAIEncoder(encoder_type)
        else:
            self.encoder = BERTEncoder(encoder_type, device=device)

        # load the rewrite candidates
        rewrite_cand_info = json.load(open(rewrite_cand_path))
        self.rewrite_cand = rewrite_cand_info["results"]
        self.context = rewrite_cand_info["context"]

        llm_class = endpoint_to_class[rewrite_cand_info["endpoint"]]
        stop = ["\n\n"]
        self.LLM = llm_class(
            rewrite_cand_info["endpoint"],
            temperature=rewrite_cand_info["temperature"],
            top_p=rewrite_cand_info["top_p"],
            max_new_tokens=rewrite_cand_info["max_new_tokens"],
            stop=stop)

        # compute the embeddings of the skill labels for these candidates
        embedding_save_path = rewrite_cand_path.split(".")[0] + f"_{encoder_type}.npy"

        # import ipdb; ipdb.set_trace()
        if os.path.exists(embedding_save_path):
            self.embeddings = np.load(embedding_save_path).astype(np.float32)
        else:
            texts = [self.rewrite_cand[k]["skill"] for k in self.rewrite_cand]
            self.embeddings = self.encoder.encode_batch(texts, batch_size=32)
            if isinstance(self.embeddings, torch.Tensor):
                self.embeddings = self.embeddings.detach().cpu().numpy().float()
            np.save(embedding_save_path, self.embeddings.astype(np.float32))
            print(f"Embeddings saved to {embedding_save_path}")

    def select(self, test_id):
        q_text = self.env.test_dataset.format_text(test_id, "CQ-K", test=False)
        input_query = ""
        input_query += self.context
        input_query += q_text
        generated = self.LLM.predict(input_query)
        generated_embedding = self.encoder.encode(generated, cpu=False)
        if isinstance(generated_embedding, torch.Tensor):
            generated_embedding = generated_embedding.detach().cpu().numpy().float()

        if self.metric == "cosine":
            similarities = np.dot(self.embeddings, generated_embedding) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(generated_embedding))
            return np.argsort(similarities)[::-1][:self.env.shot_number]
        elif self.metric == "euclidean":
            distances = np.linalg.norm(self.embeddings - generated_embedding, axis=1)
            return np.argsort(distances)[:self.env.shot_number]
        else:
            raise ValueError(f"Unknown metric: {self.metric}")