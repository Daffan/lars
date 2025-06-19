import os
import json
import numpy as np
import torch

from src.data import data_loader_dict
from src.envs.encoders import BERTEncoder, OpenAIEncoder

def compute_embeddings(task, split, encoder_type, format, encoder=None, test=False):
    assert task in data_loader_dict.keys(), f"Invalid task {task}"
    dataset = data_loader_dict[task](split=split, shuffle=False)
    
    texts = [dataset.format_text(id, format, test=test) for id in range(len(dataset))]
    if encoder is None:
        if encoder_type in ["text-embedding-ada-002"]:
            encoder = OpenAIEncoder(encoder_type)
        else:
            encoder = BERTEncoder(model_config=encoder_type)
    embedding_save_folder = f"results/embeddings/{encoder_type}/{task}"
    os.makedirs(embedding_save_folder, exist_ok=True)
    is_test = "_test" if test else ""
    embedding_save_path = f"{embedding_save_folder}/{split}_{format}{is_test}.npy"

    label = np.ones(len(dataset))
    if task == "zero-shot-cot":
        label = np.array(dataset.get_correct_labels())

    if os.path.exists(embedding_save_path):
        corpus_embeddings = np.load(embedding_save_path).astype(np.float32)
        return corpus_embeddings, (texts, label)
    else:
        corpus_embeddings = encoder.encode_batch(texts, batch_size=32)
        if isinstance(corpus_embeddings, torch.Tensor):
            corpus_embeddings = corpus_embeddings.detach().cpu().numpy().float()
        np.save(embedding_save_path, corpus_embeddings.astype(np.float32))
        return corpus_embeddings.astype(np.float32), (texts, label)