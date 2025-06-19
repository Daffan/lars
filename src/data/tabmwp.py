import json
from typing import List, Tuple
import re
import random

import numpy as np

from src.data.data_loader import DataLoader

class TableMWPDataLoader(DataLoader):
    def __init__(self, data_root: str="assets/dataset/tablemwp", split: str="test1k", shuffle: bool = False, seed: int = None):
        super().__init__(data_root, split, shuffle, seed)
        self.data_root = data_root
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.data = json.load(open(f"{data_root}/problems_{split}.json", "r"))
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
            "CQ-K", # context + question + skill
            "Q",    # question
            "Q-S", # question + step-by-step solution + answer
        ]
        if test is True, then the tests following the '-' syntax will be replaced
        with a prefix to inccur the LLM to generate text
        """
        if p is None:
            pid = self.pids[id]
            p = self.data[pid]
        steps = [s for s in p['solution'].split("\n") if len(s) > 0]
        solution = " ".join(steps)
        answer = p['answer']
        skill = p['skill'] if 'skill' in p else ""
        elements = {
            "C": f"Table: [TITLE]: {p['table_title']}\n{p['table']}",
            "Q": "Question: %s" % self._get_question_text(p),
            "S": f"Solution: {solution} The answer is {answer}",
            "K": f"Skill: {skill}",
        }

        if "-"  in format:
            input_format, output_format = format.split("-")
        else:
            input_format, output_format = format, ""

        input_text = "\n".join([elements[l] for l in input_format])
        if not test:
            output_text = "\n".join([elements[l] for l in output_format])
        else:
            output_text = "Solution: "  #  "Solution: "  # this is the text used by PromptpG

        # Prompt text
        text = input_text + "\n" + output_text
        text = text.replace("  ", " ").strip()

        return text
    
    def verify(self, id: int, text: str) -> Tuple[bool, Tuple[str, str]]:
        """ Return (correct, (prediction, answer)) """
        p = self.data[self.pids[id]]
        answer_norm = self._normalize_answer(p["answer"], p["unit"])
        prediction = self._extract_prediction(text, p["choices"])
        prediction_norm = self._normalize_answer(prediction, p["unit"])
        return prediction_norm.lower() == answer_norm.lower(), (prediction_norm, answer_norm)

    def _get_question_text(self, problem, option_inds=["A", "B", "C", "D", "E", "F"]):
        question = problem['question']

        unit = problem['unit']
        if unit and len(unit) > 0:
            question = f"{question} (Unit: {unit})"

        choices = problem['choices']
        if choices and len(choices) > 0:
            choice_list = []
            for i, c in enumerate(choices):
                choice_list.append("({}) {}".format(option_inds[i], c))
            options = " ".join(choice_list)
            #print(options)
            question = f"{question}\nOptions: {options}"

        return question
    
    def _normalize_answer(self, text, unit):
        # ["1,000", "123", "3/4", "56.456", "$56.4", "-3", "-10.02", "-3/2"]

        text = re.sub("^[\$]", "", text)
        text = re.sub("[\,\.\,\/]$", "", text)
        result = re.match("^[-+]?[\d,./]+$", text)

        if result is not None:
            # is number?
            text = text.replace(",", "")
            result = re.match("[-+]?\d+$", text)
            try:
                if result is not None:
                    number = int(text)
                elif "/" in text:
                    nums = text.split("/")
                    number = round(float(nums[0]) / float(nums[1]), 3)
                else:
                    number = round(float(text), 3)
                number = str(number)
                number = re.sub(r"\.[0]+$", "", number)
                return number
            except:
                return text
        else:
            # is text
            if unit:
                text = text.replace(unit, "").strip()
            return text
        
    def _extract_prediction(self, output, options, option_inds=["A", "B", "C", "D", "E", "F"]):
        # $\\frac{16}{95}$ -> 16/95
        output = re.sub(r"\$?\\frac\{([\d\.\,\-]+)\}\{([\d\.\,]+)\}\$?", r"\1/\2", output)

        output = re.sub(r"(?<![AP]\.M)\.$", "", output)
        output = re.sub(r"(?<=\d)[\=](?=[\-\$\d])", " = ", output)
        output = re.sub(r"\u2212", "-", output)

        ## Multi-choice questions
        if options:
            patterns = [
                r'^\(([A-Za-z])\)$',  # "(b)", "(B)"
                r'^([A-Za-z])$',  # "b", "B"
                r'^([A-Za-z]). ',  # "b", "B"
                r'[Th]he answer is ([A-Z])',  # "The answer is B"
                r'^\(([A-Za-z])\) [\s\S]+$',  # "(A) XXXXX"
                r'[Th]he answer is \(([A-Za-z])\) [\s\S]+$',  # "The answer is (B) XXXXX."
            ]

            # have "X" in the output
            for p in patterns:
                pattern = re.compile(p)
                res = pattern.findall(output)
                if len(res) > 0:
                    pred = res[0].upper()  # e.g., "B"
                    if pred in option_inds:
                        ind = option_inds.index(pred)  # 1
                        if ind >= len(options):
                            ind = random.choice(range(len(options)))
                        predition = options[ind]
                        return predition

            # find the most similar options
            scores = [self._score_string_similarity(x, output) for x in options]
            max_idx = int(np.argmax(scores))  # json does not recognize NumPy data types
            predition = options[max_idx]
            return predition

        else:
            ## free_text QA problems, numeric answer
            patterns = [
                # r'^\([A-Za-z]\) ([\s\S]+)$', # "(A) XXXXX"
                # r'[Th]he answer is \([A-Za-z]\) ([\s\S]+)$', # "The answer is (B) XXXXX."
                r'[Th]he answer is ([\s\S]+)$',  # "The answer is XXXXX.",
                r'[Th]he table shows that ([\d\$\.\,\/\:]+) ',
                r' = ([\d\$\.\,\/\:]+)',  # "= $1.40"
                r'(?<= be| is) ([\-\d\$\.\,\/\:]{0,}[\d]+)',  # "will be $1.40"
                r'(?<= are| was) ([\-\d\$\.\,\/\:]{0,}[\d]+)',  # "are $1.40"
                r'(?<= were) ([\-\d\$\.\,\/\:]{0,}[\d]+)',  # "are $1.40"
                r' ([\d\$\.\,\/\:]+ [AP]\.M\.)',  # 7:25 P.M.
                r'([\-\d\$\.\,\/\:]{0,}[\d]+)',  # 14.5
            ]

            for p in patterns:
                pattern = re.compile(p)
                res = pattern.findall(output)
                if len(res) > 0:
                    predition = res[-1].strip()
                    if predition.endswith(".") and ".M." not in predition:
                        predition = predition[:-1]
                    return predition

        return output
    
    def _score_string_similarity(self, str1, str2):
        if str1 == str2:
            return 2.0
        if " " in str1 or " " in str2:
            str1_split = str1.split(" ")
            str2_split = str2.split(" ")
            overlap = list(set(str1_split) & set(str2_split))
            return len(overlap) / max(len(str1_split), len(str2_split))
        else:
            if str1 == str2:
                return 1.0
            else:
                return 0.0