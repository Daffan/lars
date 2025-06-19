from typing import List, Tuple, Optional, Any

import gymnasium as gym
import numpy as np

from src.llm.base import BaseLLM
from src.envs.encoders import BaseEncoder

class CoTEnv(gym.Env):
    def __init__(self, analyzer: BaseLLM, executor: Optional[BaseLLM], encoder: BaseEncoder, max_num_step=5):
        """
        An CoT agent selects prompt from a list of prompts. Based on the prompt,
        the env first queries analyzer for the next step thought, then queries executor to perform
        the thought. Finally, the encoder computes the texts of all history into an embedding.
        args:
            analyzer: a LLM of text analyzer
            executor: a LLM of text executor
            encoder: a text encoder of texts into embedding
        """
        self.analyzer = analyzer
        self.executor = executor
        self.encoder = encoder

        self.analyzer_gens = []
        self.executor_gens = []

        # for the purpose of logging
        self.analyzer_prompts = []
        self.analyzer_gen_full = []
        self.executor_prompts = []
        self.executor_gen_full = []
        
        self.max_num_step = max_num_step
        self.curr_num_step = None
        
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()  # BERT embedding space

    def _get_action_space(self):
        raise NotImplementedError

    def _get_observation_space(self):
        raise NotImplementedError

    def seed(self, seed):
        pass

    def step(self, action):
        self.curr_num_step += 1
        analyzer_prompt = self._prepare_analyzer_prompt(self.analyzer_gens, self.executor_gens, action)
        self.analyzer_prompts.append(analyzer_prompt)

        # raw text from analyzer LLM
        generated_text = self.analyzer.predict(analyzer_prompt)
        # post-processing the text
        analyzer_thought = self._post_propcessing_analyzer_thought(generated_text)
        while analyzer_thought is None:
            generated_text = self.analyzer.predict(analyzer_prompt)
            analyzer_thought = self._post_propcessing_analyzer_thought(generated_text)
        self.analyzer_gen_full.append(generated_text)
        self.analyzer_gens.append(analyzer_thought)

        # raw text from executor LLM
        if self.executor is not None:
            executor_prompt = self._prepare_executor_prompt(self.analyzer_gens, self.executor_gens)
            self.executor_prompts.append(executor_prompt)
            # post-processing the text
            generated_text = self.executor.predict(executor_prompt)
            executor_thought = self._post_propcessing_executor_thought(generated_text)
            while executor_thought is None:
                generated_text = self.executor.predict(executor_prompt)
                executor_thought = self._post_propcessing_executor_thought(generated_text)
            self.executor_gen_full.append(generated_text)
            self.executor_gens.append(executor_thought)

        done = self._check_termination(self.analyzer_gens, self.executor_gens, action)
        truncated = self.curr_num_step >= self.max_num_step
        done = done or truncated

        return (
            self._compute_embedding(self.analyzer_gens, self.executor_gens),
            self._compute_reward(self.analyzer_gens, self.executor_gens, action, done),
            done, truncated, self._get_info(done),
        )
    
    def _post_propcessing_analyzer_thought(self, generated_text):
        raise NotImplementedError

    def _post_propcessing_executor_thought(self, generated_text):
        raise NotImplementedError

    def _prepare_analyzer_prompt(self, analyzer_gens: List[str], executor_gens: List[str], action) -> str:
        raise NotImplementedError

    def _prepare_executor_prompt(self, analyzer_gens: List[str], executor_gens: List[str]) -> str:
        raise NotImplementedError
    
    def _compute_embedding(self, analyzer_gens: List[str], executor_gens: List[str]) -> np.ndarray:
        raise NotImplementedError

    def _compute_reward(self, analyzer_gens: List[str], executor_gens: List[str], action: int, done: bool) -> float:
        raise NotImplementedError

    def _check_termination(self, analyzer_gens: List[str], executor_gens: List[str], action: Any) -> bool:
        raise NotImplementedError
    
    def _get_info(self, done: bool) -> dict:
        return dict(
            analyzer_prompts=self.analyzer_prompts,
            analyzer_gen_full=self.analyzer_gen_full,
            executor_prompts=self.executor_prompts,
            executor_gen_full=self.executor_gen_full,
            analyzer_gens=self.analyzer_gens,
            executor_gens=self.executor_gens,
        )

    def reset(self, seed=None):
        self.analyzer_gens = []
        self.executor_gens = []
        self.curr_num_step = 0

        self.analyzer_prompts = []
        self.analyzer_gen_full = []
        self.executor_prompts = []
        self.executor_gen_full = []
        return self._compute_embedding(self.analyzer_gens, self.executor_gens), self._get_info(done=False)

    def render(self, mode="human"):
        pass

    def close(self):
        pass
    
    def seed(self, seed=None):
        pass