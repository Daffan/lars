from typing import Optional, List

import numpy as np
import gymnasium as gym 

from src.envs.base import CoTEnv
from src.envs.encoders import BaseEncoder
from src.llm.base import BaseLLM
from src.data.data_loader import DataLoader

STEP_DELIMITER = "\\n"

class CoTDatasetEnv(CoTEnv):
    """ LLM environment for CoT prompting. It only has analyzer LLM.
    args:
        LLM: a LLM of text analyzer (e.g., reasoning question answering)
        encoder: a text encoder of texts into embedding
        test_dataset: a dataset of test problems
        cand_dataset: a dataset of candidate examples
        shot_number: the number of in-context examples
        action_probability: whether the action is the probability of selecting each candidate,
                            otherwise it is the index of the selected candidate
        format: the format of the prompt, e.g., "CQ-S" for concatenating the candidate and the question with a separator, then followed by the reasoning steps
        max_num_step: the maximum number of steps to perform, if it is 1, then it is single-step selection (generating the entire reasoning chain)
        step_delimiter: the delimiter of the steps
    """
    def __init__(
        self,
        LLM: BaseLLM,
        encoder: BaseEncoder,
        test_dataset: DataLoader,
        cand_dataset: DataLoader,
        shot_number: int=2,
        action_probability: bool=True,
        format: str="CQ-S",
        max_num_step=5,
        step_delimiter="\n"):
        
        self.test_dataset = test_dataset
        self.cand_dataset = cand_dataset
        self.shot_number = shot_number
        self.action_probability = action_probability
        self.format = format
        self.step_delimiter = step_delimiter

        super().__init__(LLM, None, encoder, max_num_step)
        self.test_id = -1

        self.shot_ids_steps = []

    def _get_action_space(self):
        return gym.spaces.Box(low=0.0, high=1.0, shape=(len(self.cand_dataset),), dtype=np.float32)
    
    def _get_observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.encoder.get_embedding_size(),))
    
    def _prepare_analyzer_prompt(self, analyzer_gens, executor_gens, action):
        # action is the probability of selecting each candidate
        if self.action_probability:
            p = action / np.sum(action)
            shot_ids = np.random.choice(range(len(self.cand_dataset)), self.shot_number, p=p, replace=False)
        else:
            shot_ids = action
        self.shot_ids_steps.append(shot_ids)
        test_id = self.test_id

        prompt = "\n\n".join([
            self.cand_dataset.build_prompt_shot(shot_ids, self.format),  # in-context examples
            self.test_dataset.build_prompt_test(test_id, "CQ-S"),  # test question
        ])

        if len(analyzer_gens) > 0:
            # if multiple steps prompting...
            prompt = prompt.replace("Q:", "Question:")
            prompt = prompt.replace("A:", "History:")
            prompt += ' ' + self.step_delimiter.join(analyzer_gens) + self.step_delimiter + "\nAnswer: Let's continue the steps.\\n"

        return prompt

    def _post_propcessing_analyzer_thought(self, generated_text):
        gt = generated_text.split("\n\n")[0]  # "\n\n" is the default delimiter between examples
        steps = [s for s in gt.split(self.step_delimiter) if len(s) > 0]  # separate steps in to a list and filter out empty strings
        if len(steps) == 0:
            print("Warning: no steps generated!")
            return None  # this is a bad call!!!

        if self.max_num_step == 1 or self.curr_num_step == self.max_num_step:
            # single-step selection
            # Or multi-step selection but the maximum number of steps is reached
            return self.step_delimiter.join(self.analyzer_gens + steps).strip()
        else:
            # multi-step selection, only use the next one generated step
            return steps[0].strip()

    def reset(self, seed=None):
        self.test_id += 1
        self.shot_ids_steps = []
        return super().reset()
        
    def _compute_embedding(self, analyzer_gens: List[str], executor_gens: List[str]) -> np.ndarray:
        # encode the current problem and the intermediate steps so far
        problem_text = self.test_dataset.build_prompt_test(self.test_id, "CQ-S")
        if len(analyzer_gens) > 0:
            problem_text += self.step_delimiter.join(analyzer_gens)
        return problem_text  # self.encoder.encode(problem_text).cpu().numpy()
    
    def _compute_reward(self, analyzer_gens: List[str], executor_gens: List[str], action: int, done: bool) -> float:
        if not done:
            return 0.0
        else:
            solution = self.step_delimiter.join(analyzer_gens)
            score, (self.answer, self.prediction) = self.test_dataset.verify(self.test_id, solution)
            return float(score)
        
    def _check_termination(self, analyzer_gens: List[str], executor_gens: List[str], action: int) -> bool:
        assert self.curr_num_step == len(analyzer_gens), f"curr_num_step ({self.curr_num_step}) is not consistent with analyzer_gens ({len(self.analyzer_gens)})"
        timeout = self.curr_num_step >= self.max_num_step
        return timeout or "The answer is" in analyzer_gens[-1]
    
    def _get_info(self, done: bool) -> dict:
        info = super()._get_info(done)
        info["test_id"] = self.test_id
        info["test_uid"] = self.test_dataset.unique_ids[self.test_id]  # need to log unique id instead of the the ids in the dataset
        info["shot_uids_steps"] = [
            [self.cand_dataset.unique_ids[i] for i in shot_ids]\
            for shot_ids in self.shot_ids_steps]
        if done:
            info["answer"] = self.answer
            info["prediction"] = self.prediction
        return info


if __name__ == "__main__":
    from src.data import data_loader_dict, AQuADataLoader, TableMWPDataLoader, GSMDataLoader, ZeroShotCoTDataLoader
    from src.envs.encoders import BERTEncoder
    from src.llm.sagemaker_endpoint import SagemakerEndpoint
    from src.llm.openai import OpenAILLM
    from src.llm import endpoint_to_class

    # perform some unit tests
    task = "cogs"  # in ["aqua", "tabmwp", "gsm8k", "spider", "strategy-qa", "mawps", "cogs"]
    test_dataset = data_loader_dict[task](split="test_reformat", shuffle=True)
    cand_dataset = data_loader_dict[task](split="train_cand1k")

    endpoint = "text-davinci-003"  # "falcon-40b-instruct", "gpt-3.5-turbo", "vicuna-13b-v1dot3", "open-llama-13b", "dolly-v2-12b", "jumpstart-dft-gpt-neox-20b" "text-davinci-003"
    temperature = 0.001
    top_p = 0.6
    max_new_tokens = 256
    llm_class = endpoint_to_class[endpoint]
    stop = ["\n\n"]
    LLM = llm_class(
        endpoint,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        stop=stop)

    encoder = BERTEncoder("microsoft/deberta-v2-xlarge", device="cpu")

    # env = CoTDatasetEnv(LLM, encoder, test_dataset, cand_dataset, shot_number=8, max_num_step=5, format="CQP-N")
    env = CoTDatasetEnv(LLM, encoder, test_dataset, cand_dataset, shot_number=16, max_num_step=1, format="CQ-S")
    
    env.reset()
    print("\n\n")
    action = np.ones(env.action_space.shape[0]) / env.action_space.shape[0]
    # action[[21, 17, 2, 6]] = 0.25
    for i in range(10):
        obs, reward, done, truncated, info = env.step(action)
        test_text = info['analyzer_prompts'][-1]
        if False:
            test_text = test_text.split('\n\n')[-1]
        print(f">>>>>>>>>> Step-{i+1} Prompt:\n{test_text}")
        print("\n")
        print("# " + info["analyzer_gen_full"][-1] + "\n\n")
        if done:
            env.reset()
            print("Episode finished after {} timesteps".format(i+1))
            print(info["shot_uids_steps"])
            print("GT answer: %s;\nPredicted answer: %s;\nreward: %.2f" %(info["answer"], info["prediction"], reward))
            break
    
    env.close()