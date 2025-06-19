import json
from typing import List, Tuple
import re
import random

import numpy as np

from src.data.data_loader import DataLoader
from src.data.utils import answer_cleansing

class GSMDataLoader(DataLoader):
    def __init__(self, data_root: str="assets/dataset/grade-school-math/grade_school_math/data", split: str="train_remain", shuffle: bool = False, seed: int = None):
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

    def format_text(self, id: int, format: str, test=False, p=None) -> str:
        """ Choice of formats: 
        [
            "CQ",   # context + question
            "CQ-S" # context + question + step-by-step solution + answer
            "Q",    # question
            "Q-S", # question + step-by-step solution + answer
            "Q-K", # question + skill
            "CQPS-NS", # context + question + previous steps + next steps
        ]
        if test is True, then the tests following the '-' syntax will be replaced
        with a prefix to inccur the LLM to generate text
        """
        if p is None:
            pid = self.pids[id]
            p = self.data[pid]
        steps = [re.sub(r"<<.*?>>", "", s) for s in p['answer'].split("\n") if len(s) > 0]
        steps[-1] = steps[-1].replace("####", "The answer is")
        solution = "\n".join(steps)
        skill = p['skill'] if 'skill' in p else ""
        elements = {
            "C": "",
            "Q": "Q: %s" %(p['question']),
            "S": f"A: Let's think step by step. \n{solution}",
            "K": f"Skill: {skill}",
        }

        if "-"  in format:
            input_format, output_format = format.split("-")
        else:
            input_format, output_format = format, ""

        input_text = "\n".join([elements[l] for l in input_format if elements[l]])
        if not test:
            output_text = "\n".join([elements[l] for l in output_format])
        else:
            output_text = "A: Let's think step by step. \n"

        # Prompt text
        text = input_text + "\n" + output_text
        text = text.replace("  ", " ").strip()

        return text

    def verify(self, id: int, text: str) -> Tuple[bool, Tuple[str, str]]:
        """ Return (correct, (prediction, answer)) """
        p = self.data[self.pids[id]]
        answer = p['answer'].split("#### ")[-1].replace(",", "")
        answer = str(round(float(answer), 2))
        prediction = answer_cleansing("gsm8k", text)
        return answer == prediction, (answer, prediction)