import json
from typing import List, Tuple
import re
import random
import os

import numpy as np

from src.data.data_loader import DataLoader
from src.data.utils import answer_cleansing
from src.utils.eval_spider import *

class SpiderDataLoader(DataLoader):
    def __init__(self, data_root: str="assets/dataset/spider", split: str="train_remain", shuffle: bool = False, seed: int = None):
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
        self.evaluator = Evaluator()
        self.kmaps = build_foreign_key_map_from_json(f"{data_root}/tables.json")

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
            "CQPS-NS", # context + question + previous steps + next steps
            "CQ-K", # context + question + skill
        ]
        if test is True, then the tests following the '-' syntax will be replaced
        with a prefix to inccur the LLM to generate text
        """
        if p is None:
            pid = self.pids[id]
            p = self.data[pid]
        skill = p["skill"] if "skill" in p else ""
        elements = {
            "C": "### Database schema:\n%s" %(p['schema']),
            "Q": "### Task: %s" %(p['question']),
            "S": "### SQL query: %s" %(p["answer"]),
            "K": "### Skills: to solve this task in the database, we need to %s" %(skill),
        }

        if "-"  in format:
            input_format, output_format = format.split("-")
        else:
            input_format, output_format = format, ""

        input_text = "\n".join([elements[l] for l in input_format if elements[l]])
        if not test:
            output_text = "\n".join([elements[l] for l in output_format])
        else:
            output_text = "### SOL query:"

        # Prompt text
        text = input_text + "\n" + output_text
        text = text.replace("  ", " ").strip()

        return text

    def verify(self, id: int, text: str) -> Tuple[bool, Tuple[str, str]]:
        db_dir = "assets/database"  # "/home/ubuntu/spider/database"
        db_name = self.data[self.pids[id]]["db_id"]
        kmap = self.kmaps[db_name]
        db = os.path.join(db_dir, db_name, db_name + ".sqlite")
        g_str = self.data[self.pids[id]]["answer"].replace("value", "1")
        p_str = text.replace("value", "1")
        schema = Schema(get_schema(db))
        g_sql = get_sql(schema, g_str)
        try:
            p_sql = get_sql(schema, p_str)
            g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
            g_sql = rebuild_sql_val(g_sql)
            g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
            p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
            p_sql = rebuild_sql_val(p_sql)
            p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)
            exact_score = self.evaluator.eval_exact_match(p_sql, g_sql)
            # exact_score = self.evaluator.eval_partial_match(p_sql, g_sql)
            return exact_score, (g_str, p_str)
        except Exception as e:
            print("Error in parsing predicted SQL!!!!!!")
            print(e)
            return 0, (g_str, p_str)
