# This script rewrite the questions into skill-based descriptions

import argparse
import random
import os
import datetime
import json

# add parent path to sys
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from src.algos.skill_discovery import Encoder, Decoder, Model, get_batched_loss
from src.data import data_loader_dict
from src.llm import endpoint_to_class
from src.envs.encoders import BERTEncoder, OpenAIEncoder
from src.utils.utils import compute_embeddings

def sample_ids_from_clustering(embeddings, cluster_centers, num_examples=3):
    # compute distance to cluster center
    distances = np.linalg.norm(embeddings[None, :, :] - cluster_centers[:, None, :], axis=-1)
    return np.argsort(distances, axis=1)[:, :num_examples]

def parse_args():
    parser = argparse.ArgumentParser()
    # user config
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--verbose', action='store_true', help='print the generations')
    # task config
    parser.add_argument('--task', type=str, default='spider')
    parser.add_argument('--split', type=str, default='train_cand1k')  # those two are the test splits
    # train config
    parser.add_argument('--endpoint', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--top_p', type=float, default=0.6)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--num_examples', type=int, default=4)
    # test config (visualize how separate are different clusters)
    parser.add_argument('--test', action='store_true', help='if true, only test the model')
    parser.add_argument('--encoder', type=str, default='microsoft/deberta-v2-xlarge')
    parser.add_argument('--test_split', type=str, default='train_cand1k')
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--plot_model', type=str, default="tsne", choices=["tsne", "pca"], help="which model to plot the embeddings")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # --- seeding ---
    if args.seed is None:
        seed = random.randint(0, 10000)
    else:
        seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = vars(args)
    cfg["seed"] = seed

    # --- load LLM ---
    llm_class = endpoint_to_class[args.endpoint]
    stop = ["\n\n"]
    LLM = llm_class(
        args.endpoint,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        stop=stop)
    
    # --- prepare annotation examples ---
    # load data
    if args.task == "cogs":
        root_folder = os.path.join("assets/dataset", "COGS") 
    elif args.task == "tabmwp":
        root_folder = os.path.join("assets/dataset", "tablemwp")
    elif args.task == "gsm8k":
        root_folder = os.path.join("assets/dataset", "grade-school-math", "grade_school_math")
    else:
        root_folder = os.path.join("assets/dataset", args.task)
    annotations = json.load(open(
        os.path.join(root_folder, "skill_knn", "annotation.json")))
    # shuffle annotations
    random.shuffle(annotations)
    
    cand_dataset = data_loader_dict[args.task](split=args.split, shuffle=False)
    
    # prepare examples in the context
    if args.task == "spider":
        context = "Generate the needed skills to solve the task on the database schema.\n\n"
    elif args.task == "cogs":
        context = "Generate the required skills to parse the following sentences.\n\n"
    elif args.task == "tabmwp":
        context = "Generate the required skills to solve the following problems based on the data from the tables\n\n"
    elif args.task == "gsm8k":
        context = "Generate the required skills to solve the following questions\n\n"
    else:
        raise NotImplementedError
    
    # using the dataset's default format text function...
    for a in annotations:
        context += cand_dataset.format_text(-999, "CQ-K", test=False, p=a)
        context += "\n\n"

    rewrited_candidates = {
        "context": context,
        "num_examples": args.num_examples,
        "seed": args.seed,
        "endpoint": args.endpoint,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "split": args.split,
        "results": {}
    }
    # timestamped saving path
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(root_folder, "skill_knn", timestamp + "_seed%d" %args.seed)
    os.makedirs(save_path, exist_ok=True)
    for i in range(len(cand_dataset)):
        input_query = ""
        input_query += context
        input_query += cand_dataset.format_text(i, "CQ-K", test=False)

        # generate the skills
        generated = LLM.predict(input_query)

        if args.verbose:
            print(input_query + generated)
        
        print(f">>>>>>>>>> Progress    {i+1} / {len(cand_dataset)} <<<<<<<<<<")
        rewrited_cand = {}
        rewrited_cand.update(cand_dataset.get_problem(i))
        rewrited_cand["skill"] = generated
        rewrited_candidates["results"][cand_dataset.unique_ids[i]] = rewrited_cand

        # save every 10 examples
        if i % 10 == 0:
            json.dump(rewrited_candidates, open(
                os.path.join(save_path, "rewrited_" + args.split + ".json"), "w"), indent=4)

    # save the rewrited candidates
    json.dump(rewrited_candidates, open(
        os.path.join(save_path, "rewrited_" + args.split + ".json"), "w"), indent=4)    