from abc import ABC, abstractmethod

import numpy as np

from src.envs.cot_env import CoTDatasetEnv

class SelectionMethod(ABC):
    def __init__(self, env: CoTDatasetEnv, seed: int=11):
        self.env = env
        self.seed = seed

    @abstractmethod
    def select(self, test_id: int) -> np.ndarray:
        raise NotImplementedError