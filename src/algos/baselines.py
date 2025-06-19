import json
import os

import numpy as np
import torch

from src.utils.utils import compute_embeddings
from src.algos.base import SelectionMethod
from src.envs.cot_env import CoTDatasetEnv
from src.envs.encoders import BaseEncoder, BERTEncoder, OpenAIEncoder
from src.algos.skill_discovery import Encoder


class RandomSelection(SelectionMethod):
    def select(self, test_id: int) -> np.ndarray:
        np.random.seed(self.seed + test_id + 111)  # this will make sure each problem has the same few-shot examples

        action_space = self.env.action_space
        p = np.ones(action_space.shape) / action_space.shape[0]
        shot_ids = np.random.choice(range(len(self.env.cand_dataset)), self.env.shot_number, p=p, replace=False)
        return shot_ids

class RepeatSelection(SelectionMethod):
    def __init__(self, env: CoTDatasetEnv, repeat_selection_path: str):
        self.env = env
        self.repeat_selection = json.load(open(repeat_selection_path))

    def select(self, test_id: int) -> np.ndarray:
        pid = self.env.test_dataset.unique_ids[test_id]
        shot_pids = self.repeat_selection["results"][pid]["shot_uids_steps"][0]
        shot_ids = [self.env.cand_dataset.unique_ids.index(shot_pid) for shot_pid in shot_pids]
        return shot_ids
    
class RetrievalQ(SelectionMethod):
    def __init__(
        self,
        env: CoTDatasetEnv,
        encoder_type: str,
        task: str,
        cand_split: str,
        seed: int=11,
        q_prompt_format: str="CQ",
        device: str="cpu",
        metric: str="cosine"):

        self.env = env
        self.encoder_type = encoder_type
        self.seed = seed
        self.q_prompt_format = q_prompt_format
        self.metric = metric
        self.device = device

        if encoder_type in ["text-embedding-ada-002", "code-search-babbage-text-001"]:
            self.encoder = OpenAIEncoder(encoder_type)
        else:
            self.encoder = BERTEncoder(encoder_type, device=device)

        self.cand_embeddings, _ = compute_embeddings(task, cand_split, encoder_type, q_prompt_format, self.encoder, test=False)

    def select(self, test_id: int) -> np.ndarray:
        # Q for retrieval
        q_text = self.env.test_dataset.format_text(test_id, self.q_prompt_format, test=False)
        q_embedding = self.encoder.encode(q_text, cpu=True)

        if self.metric == "cosine":
            # compute the cosine similarity between the query and the candidates
            similarities = np.dot(self.cand_embeddings, q_embedding) / (np.linalg.norm(self.cand_embeddings, axis=1) * np.linalg.norm(q_embedding))
            return np.argsort(similarities)[::-1][:self.env.shot_number]
        elif self.metric == "euclidean":
            distances = np.linalg.norm(self.cand_embeddings - q_embedding, axis=1)
            return np.argsort(distances)[:self.env.shot_number]
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        

class RetrievalRSD(SelectionMethod):
    def __init__(
        self,
        env: CoTDatasetEnv,
        encoder_type: str,
        skill_encoder_path: str,
        task: str,
        cand_split: str,
        seed: int=11,
        device: str="cpu",
        use_pi: bool=False,
        metric: str="cosine"):

        self.env = env
        self.encoder_type = encoder_type
        self.seed = seed
        self.metric = metric

        self.skill_encoder_path = skill_encoder_path
        self.device = device
        self.use_pi = use_pi

        if encoder_type in ["text-embedding-ada-002", "code-search-babbage-text-001"]:
            self.encoder = encoder = OpenAIEncoder(encoder_type)
        else:
            self.encoder = encoder = BERTEncoder(encoder_type, device=device)

        self.skill_config, self.q_encoder, self.qs_encoder = self._load_skill_encoders()
        prompt_format = self.skill_config["prompt_format"]
        q_prompt_format, s_prompt_format = prompt_format.split("-")[0], prompt_format.split("-")[1]

        if self.skill_config["qs_as_whole"]:
            cand_embeddings, _ = compute_embeddings(task, cand_split, encoder_type, prompt_format, encoder, test=False)
        else:
            cand_q_embeddings, _ = compute_embeddings(task, cand_split, encoder_type, q_prompt_format, encoder, test=False)
            cand_s_embeddings, _ = compute_embeddings(task, cand_split, encoder_type, s_prompt_format, encoder, test=False)
            cand_embeddings = np.concatenate([cand_q_embeddings, cand_s_embeddings], axis=-1)
        self.cand_skill_embeddings = self.qs_encoder(torch.from_numpy(cand_embeddings).to(device))[0].detach().cpu().numpy()

    def _load_skill_encoders(self):
        skill_config = json.load(open(os.path.join(os.path.dirname(self.skill_encoder_path), "config.json")))
        q_encoder = Encoder(
            input_dim=self.encoder.get_embedding_size(),
            hidden_dim=skill_config["hidden_dim"],
            hidden_layer=skill_config["hidden_layer"],
            latent_dim=skill_config["latent_dim"]).to(self.device)
        state_dict = torch.load(self.skill_encoder_path)
        state_dict_q_encoder = {}
        for k in state_dict.keys():
            if not self.use_pi:
                if k.startswith("q_encoder."):
                    state_dict_q_encoder[k[10:]] = state_dict[k]
            else:
                if k.startswith("pi."):
                    state_dict_q_encoder[k[3:]] = state_dict[k]
        q_encoder.load_state_dict(state_dict_q_encoder)
        q_encoder.eval()

        qs_input_size = self.encoder.get_embedding_size() if skill_config["qs_as_whole"] else self.encoder.get_embedding_size() * 2
        qs_encoder = Encoder(
            input_dim=qs_input_size,
            hidden_dim=skill_config["hidden_dim"],
            hidden_layer=skill_config["hidden_layer"],
            latent_dim=skill_config["latent_dim"]).to(self.device)
        state_dict_qs_encoder = {}
        for k in state_dict.keys():
            if k.startswith("qs_encoder."):
                state_dict_qs_encoder[k[11:]] = state_dict[k]
        qs_encoder.load_state_dict(state_dict_qs_encoder)
        qs_encoder.eval()

        return skill_config, q_encoder, qs_encoder

    def select(self, test_id: int) -> np.ndarray:
        # Q from retrieval
        q_prompt_format = self.skill_config["prompt_format"].split("-")[0]
        q_text = self.env.test_dataset.format_text(test_id, q_prompt_format, test=False)
        q_embedding = self.encoder.encode(q_text, cpu=False)
        if isinstance(q_embedding, np.ndarray):
            q_embedding = torch.from_numpy(q_embedding.astype(np.float32)).to(next(self.q_encoder.parameters()).device)
        q_skill_embedding = self.q_encoder(q_embedding)[0].detach().cpu().numpy()

        # compute the cosine similarity between the query and the candidates
        similarities = np.dot(self.cand_skill_embeddings, q_skill_embedding) / (np.linalg.norm(self.cand_skill_embeddings, axis=1) * np.linalg.norm(q_skill_embedding))
        return np.argsort(similarities)[::-1][:self.env.shot_number]