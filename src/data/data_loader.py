from typing import List, Tuple
from abc import ABC, abstractmethod

class DataLoader(ABC):
    @abstractmethod
    def __init__(self, data_root: str, split: str, shuffle: bool = False, seed: int = None):
        pass

    @abstractmethod
    def format_text(self, ids: int, format: str, test=False) -> List[str]:
        pass

    @abstractmethod
    def verify(self, id: int, solution: str) -> Tuple[bool, Tuple[str, str]]:
        pass

    def build_prompt_shot(self, shot_ids: List[int], format: str) -> str:
        texts = []

        for shot_id in shot_ids:
            texts.append(self.format_text(shot_id, format, test=False))

        prompt = "\n\n".join(texts)
        return prompt
    
    def build_prompt_test(self, test_id: int, format: str) -> str:
        return self.format_text(test_id, format, test=True)
    
    @property
    def unique_ids(self) -> List:
        # id to unique ids
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def get_problem(self, id: int) -> dict:
        raise NotImplementedError