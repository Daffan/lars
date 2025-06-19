import json
from typing import List, Tuple
import re
import random

import numpy as np

from src.data.data_loader import DataLoader
from src.data.utils import answer_cleansing

class AQuADataLoader(DataLoader):
    def __init__(self, data_root: str="assets/dataset/AQuA", split: str="train_remain", shuffle: bool = False, seed: int = None):
        super().__init__(data_root, split, shuffle, seed)
        self.data_root = data_root
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.data = json.load(open(f"{data_root}/{split}.json", "r"))
        self.pids = list(self.data.keys())
        if shuffle:
            np.random.shuffle(self.pids)

    def get_problem(self, id: int) -> dict:
        return self.data[self.pids[id]]

    @property
    def unique_ids(self) -> List:
        return self.pids

    def __len__(self):
        return len(self.data)

    def format_text(self, id: int, format: str, test=False) -> str:
        """ Choice of formats: 
        [
            "CQ",   # context + question
            "CQ-S" # context + question + step-by-step solution + answer
            "Q",    # question
            "Q-S", # question + step-by-step solution + answer
        ]
        if test is True, then the tests following the '-' syntax will be replaced
        with a prefix to inccur the LLM to generate text
        """
        pid = self.pids[id]
        p = self.data[pid]
        steps = [s for s in p['rationale'].split("\n") if len(s) > 0]
        solution = ".\n".join(steps[:-1])
        Question = f"Q: {p['question']}"
        if 'options' in p.keys():
            Question += f" Answer Choices: {', '.join(p['options'])}"
        elements = {
            "C": "",
            "Q": Question,
            "S": f"A: Let's think step by step.\n{solution}\nThe answer is {p['correct']}",
        }

        if "-"  in format:
            input_format, output_format = format.split("-")
        else:
            input_format, output_format = format, ""

        input_text = "\n".join([elements[l] for l in input_format if elements[l]])
        if not test:
            output_text = "\n".join([elements[l] for l in output_format])
        else:
            output_text = "A: Let's think step by step. "

        # Prompt text
        text = input_text + "\n" + output_text
        text = text.replace("  ", " ").strip()

        return text

    def verify(self, id: int, text: str) -> Tuple[bool, Tuple[str, str]]:
        """ Return (correct, (prediction, answer)) """
        p = self.data[self.pids[id]]
        answer = p['correct']
        prediction = answer_cleansing("aqua", text)
        return answer == prediction, (answer, prediction)


class ZeroShotCoTDataLoader(AQuADataLoader):
    def __init__(self, data_root: str = "assets/dataset/zero_shot_cot", split: str = "train_remain", shuffle: bool = False, seed: int = None):
        super().__init__(data_root, split, shuffle, seed)

    def verify(self, id: int, text: str) -> Tuple[bool, Tuple[str, str]]:
        """ Return (correct, (prediction, answer)) """
        p = self.data[self.pids[id]]
        if p['task'] in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
            answer = str(round(float(p['correct']), 2))
        else:
            answer = p['correct']
        prediction = answer_cleansing(p['task'], text)
        return answer == prediction, (answer, prediction)
    
    def get_correct_labels(self) -> List[int]:
        return [int(self.data[self.pids[id]]['correct'] == self.data[self.pids[id]]['pred_ans']) for id in range(len(self))]