from transformers import AutoTokenizer, AutoModelForTokenClassification, DebertaV2Model
import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np

from src.utils.utils import compute_embeddings
from src.algos.base import SelectionMethod
from src.envs.cot_env import CoTDatasetEnv
from src.data import data_loader_dict
from src.envs.encoders import BaseEncoder, BERTEncoder, OpenAIEncoder

class policy_network(nn.Module):

    def __init__(self,
                 model_config="bert-base-uncased",
                 add_linear=False,
                 embedding_size=128,
                 freeze_encoder=True) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_config)
        print("model_config:", model_config)
        if model_config == "bert-base-uncased":
            self.model = AutoModelForTokenClassification.from_pretrained(model_config)
        elif model_config == "microsoft/deberta-v2-xlarge":
            self.model = DebertaV2Model.from_pretrained(model_config)

        # Freeze transformer encoder and only train the linear layer
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        if add_linear:
            # Add an additional small, adjustable linear layer on top of BERT tuned through RL
            self.embedding_size = embedding_size
            self.linear = nn.Linear(self.model.config.hidden_size,
                                    embedding_size)  # 768 for bert-base-uncased, distilbert-base-uncased
        else:
            self.linear = None

    def forward(self, input_list):
        input = self.tokenizer(input_list, truncation=True, padding=True, return_tensors="pt").to(self.model.device)
        # print(f"input: {input}")
        output = self.model(**input, output_hidden_states=True)
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]
        # Get [CLS] hidden states
        sentence_embedding = last_hidden_states[:, 0, :]  # len(input_list) x hidden_size
        # print(f"sentence_embedding: {sentence_embedding}")

        if self.linear:
            sentence_embedding = self.linear(sentence_embedding)  # len(input_list) x embedding_size

        return sentence_embedding


class PromptPG(SelectionMethod):
    def __init__(
        self,
        env: CoTDatasetEnv,
        ckpt_path: str,
        encoder_type: str,
        task: str,
        cand_split: str,
        seed: int=11,
        device: str="cpu"):

        super().__init__(env, seed)
        self.model = policy_network(
            model_config="bert-base-uncased",
            add_linear=True,
            embedding_size=128,
            freeze_encoder=True
        ).to(device)
        self.model.linear.load_state_dict(torch.load(ckpt_path))

        self.env = env
        self.encoder_type = encoder_type
        self.seed = seed
        self.device = device

        dataset = data_loader_dict[task](split=cand_split, shuffle=False)
        texts = [dataset.format_text(id, "CQ", test=True) for id in range(len(dataset))]

        # need to make this computing batched
        batch_size = 32
        self.cand_embeddings = []
        for i in range(0, len(texts), batch_size):
            self.cand_embeddings.append(self.model(texts[i:i+batch_size]))
        self.cand_embeddings = torch.cat(self.cand_embeddings, dim=0)

    def select(self, test_id: int) -> np.ndarray:
        q_text = self.env.test_dataset.format_text(test_id, "CQ", test=True)
        q_embeddings = self.model([q_text])
        scores = F.softmax(torch.mm(q_embeddings, self.cand_embeddings.t()), dim=1)[0].detach().cpu().numpy()
        return np.argsort(scores)[::-1][:self.env.shot_number]



