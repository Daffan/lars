from abc import ABC, abstractmethod
from typing import List

class BaseLLM(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, prompt: str) -> str:
        pass


class FixedTextLLM(BaseLLM):
    def __init__(self, texts: list[str]):
        super().__init__()
        assert len(texts) > 0
        self.texts = texts
        self.curr_idx = 0
        self.ended = False

    def predict(self, prompt: str) -> str:
        generated_text = self.texts[self.curr_idx]
        if not self.ended:
            self.ended = (self.curr_idx == len(self.texts) - 1)
        self.curr_idx = (self.curr_idx + 1) % len(self.texts)
        return generated_text
    
    def end_of_texts(self) -> bool:
        return self.ended
    

class CopyLLM(BaseLLM):
    def predict(self, prompt: str) -> str:
        return prompt