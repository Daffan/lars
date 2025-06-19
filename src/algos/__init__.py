from src.envs.cot_env import CoTDatasetEnv
from src.envs.encoders import BaseEncoder
from src.algos.base import SelectionMethod
from src.algos.baselines import RandomSelection, RetrievalQ, RetrievalRSD, RepeatSelection

def init_selection_method(
    method: str,
    env: CoTDatasetEnv,
    encoder: BaseEncoder,
    task: str,
    cand_split: str,
    seed: int=11,
    q_prompt_format: str="CQ",
    metric: str="cosine",
    device: str="cpu",
    use_pi: bool=False,
    skill_encoder_path: str=None,
    rewrite_cand_path: str=None,
    prompt_pg_ckpt: str=None,
    repeat_selection_path: str=None
) -> SelectionMethod:
    if method == "random":
        return RandomSelection(env, seed)
    elif method == "retrieval_q":
        return RetrievalQ(env, encoder, task, cand_split, seed, q_prompt_format, device, metric)
    elif method == "retrieval_rsd":
        return RetrievalRSD(env, encoder, skill_encoder_path, task, cand_split, seed, device, use_pi, metric)
    elif method == "prompt_pg":
        from src.algos.prompt_pg import PromptPG
        return PromptPG(env, prompt_pg_ckpt, encoder, task, cand_split, seed, device)
    elif method == "skill_knn":
        from src.algos.skill_knn import SkillKNN
        return SkillKNN(
            env, encoder, task, 
            'gpt-3.5-turbo', rewrite_cand_path, device, metric, seed
        )
    elif method == "repeat":
        return RepeatSelection(env, repeat_selection_path)
    else:
        raise ValueError(f"Unknown selection method: {method}")