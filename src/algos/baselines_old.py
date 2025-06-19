from typing import List
import numpy as np
import torch

from src.envs.cot_env import CoTDatasetEnv

def uniform_per_problem(env: CoTDatasetEnv, id: int, gens: List[str], seed: int) -> np.ndarray:
    # uniformly sampled cands per problem
    action_space = env.action_space
    p = np.ones(action_space.shape) / action_space.shape[0]

    # this will make sure each problem has the same few-shot examples
    np.random.seed(seed + id + 111)
    shot_ids = np.random.choice(range(len(env.cand_dataset)), env.shot_number, p=p, replace=False)
    return shot_ids
    # p_new = np.zeros(action_space.shape)
    # p_new[shot_ids] = 1.0 / env.shot_number
    # return p_new

def diversity_based(env: CoTDatasetEnv, id: int, gens: List[str], seed: int, center_ids: List[int]) -> np.ndarray:
    # use the cluster centers as the few-shot examples
    assert env.shot_number == len(center_ids)
    return center_ids
""" 
    action_space = env.action_space
    p = np.zeros(action_space.shape, dtype=np.float32)
    p[center_ids] = 1.0 / len(center_ids)
    return p
 """
def uniform_per_step(env: CoTDatasetEnv, id: int, gens: List[str], seed: int) -> np.ndarray:
    # uniformly sampled cands per step
    action_space = env.action_space
    p = np.ones(action_space.shape) / action_space.shape[0]
    shot_ids = np.random.choice(range(len(env.cand_dataset)), env.shot_number, p=p, replace=False)
    return shot_ids

def retrieval_q_euclidean(env: CoTDatasetEnv, id: int, gens: List[str], seed: int, cand_embeddings, encoder) -> np.ndarray:
    # Q from retrieval
    q_text = env.test_dataset.format_text(id, "CQ", test=False)
    q_embedding = encoder.encode(q_text)
    if isinstance(q_embedding, torch.Tensor):
        q_embedding = q_embedding.detach().cpu().numpy()

    # compute the distance between the query and the candidates
    distances = np.linalg.norm(cand_embeddings - q_embedding, axis=1)
    return np.argsort(distances)[:env.shot_number]

    # compute the probability with nearest two positive
    # p = np.zeros(len(cand_embeddings))
    # p[np.argsort(distances)[:env.shot_number]] = 1.0 / env.shot_number
    # return p

def retrieval_q_cosine(env: CoTDatasetEnv, id: int, gens: List[str], seed: int, cand_embeddings, encoder) -> np.ndarray:
    # Q from retrieval
    q_text = env.test_dataset.format_text(id, "CQ", test=False)
    q_embedding = encoder.encode(q_text)
    if isinstance(q_embedding, torch.Tensor):
        q_embedding = q_embedding.detach().cpu().numpy()

    # compute the cosine similarity between the query and the candidates
    similarities = np.dot(cand_embeddings, q_embedding) / (np.linalg.norm(cand_embeddings, axis=1) * np.linalg.norm(q_embedding))
    return np.argsort(similarities)[::-1][:env.shot_number]
    # compute the probability with largest two as positive
    # p = np.zeros(len(cand_embeddings))
    # p[np.argsort(similarities)[::-1][:env.shot_number]] = 1.0 / env.shot_number
    # return p

def retrieval_q_skill_cosine(env: CoTDatasetEnv, id: int, gens: List[str], seed: int, cand_skill_embeddings, encoder, q_encoder) -> np.ndarray:
    # Q from retrieval
    q_text = env.test_dataset.format_text(id, "CQ", test=False)
    q_embedding = encoder.encode(q_text, cpu=False)
    if isinstance(q_embedding, np.ndarray):
        q_embedding = torch.from_numpy(q_embedding.astype(np.float32)).to(next(q_encoder.parameters()).device)
    q_skill_embedding = q_encoder(q_embedding)[0].detach().cpu().numpy()

    # compute the cosine similarity between the query and the candidates
    similarities = np.dot(cand_skill_embeddings, q_skill_embedding) / (np.linalg.norm(cand_skill_embeddings, axis=1) * np.linalg.norm(q_skill_embedding))
    return np.argsort(similarities)[::-1][:env.shot_number]
    # compute the probability with largest two as positive
    # p = np.zeros(len(cand_skill_embeddings))
    # p[np.argsort(similarities)[::-1][:env.shot_number]] = 1.0 / env.shot_number
    # return p

def step_retrieval_q_skill_cosine(env: CoTDatasetEnv, id: int, gens: List[str], seed: int, cand_skill_embeddings, encoder, q_encoder) -> np.ndarray:
    # Q from retrieval
    q_text = env.test_dataset.format_text(id, "CQ", test=True)
    if len(gens) > 0:
        q_text = q_text.replace("Q:", "Question:")
        q_text = q_text.replace("A:", "History:")
        q_text += ' ' + "\\n".join(gens) + "\\n" + "\nAnswer: Let's continue the steps.\\n"

    q_embedding = encoder.encode(q_text, cpu=False)
    q_skill_embedding = q_encoder(q_embedding)[0].detach().cpu().numpy()

    # compute the cosine similarity between the query and the candidates
    similarities = np.dot(cand_skill_embeddings, q_skill_embedding) / (np.linalg.norm(cand_skill_embeddings, axis=1) * np.linalg.norm(q_skill_embedding))
    return np.argsort(similarities)[::-1][:env.shot_number]