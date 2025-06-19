import argparse
import json
import os
import random
import numpy as np
import torch

# add parent path to sys
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import data_loader_dict
from src.envs.cot_env import CoTDatasetEnv
from src.envs.encoders import DummyEncoder
from src.llm import endpoint_to_class
from src.algos import init_selection_method

def parse_args():
    parser = argparse.ArgumentParser()
    # user config
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--verbose', action='store_true', help='print the generations')
    parser.add_argument('--cpu', action='store_true')
    # task config
    parser.add_argument('--task', type=str, default='tablemwp')
    parser.add_argument('--split', type=str, default='test_reformat')  # test splits
    parser.add_argument('--endpoint', type=str, default='falcon-40b-instruct')
    # prompting config
    parser.add_argument('--encoder', type=str, default='microsoft/deberta-v2-xlarge', 
        choices=[
            'microsoft/deberta-v2-xlarge',
            'sentence-transformers/all-MiniLM-L6-v2',
            'text-embedding-ada-002',
            'code-search-babbage-text-001'
        ])
    parser.add_argument('--cand_split', type=str, default='train_cand1k', help='the split to sample the example problems from')
    parser.add_argument('--shot_number', type=int, default=2)
    parser.add_argument('--prompt_format', type=str, default='CQ-S', help='format of the exmaples to prompt the LLM')
    # in-context examples selection strategy
    parser.add_argument(
        '--selection', type=str, default='random',
        choices=[
            'random',
            'retrieval_q',
            'retrieval_rsd',
            'prompt_pg',
            'skill_knn',
            'repeat',  # repeat an existing selection
        ]
    )
    parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--skill_encoder', type=str, default=None)
    parser.add_argument('--rewrite_cand_path', type=str, default=None)
    parser.add_argument('--repeat_selection_path', type=str, default=None)
    parser.add_argument('--prompt_pg_ckpt', type=str, default=None)
    parser.add_argument('--use_pi', action='store_true')
    # LLM config
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.001)
    parser.add_argument('--top_p', type=float, default=0.6)

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

    # --- load data ---
    assert args.task in data_loader_dict.keys(), f"Invalid task {args.task}"
    test_dataset = data_loader_dict[args.task](split=args.split, shuffle=False)
    cand_dataset = data_loader_dict[args.task](split=args.cand_split, shuffle=False)
    test_uids = test_dataset.unique_ids.copy()

    # --- load LLM ---
    llm_class = endpoint_to_class[args.endpoint]
    stop = ["\n\n"]
    LLM = llm_class(
        args.endpoint,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        stop=stop)
    
    # --- load encoder ---
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # --- init environment ---
    env = CoTDatasetEnv(
        LLM, DummyEncoder(), test_dataset, cand_dataset, action_probability=False,
        shot_number=args.shot_number, max_num_step=1, format="CQ-S")
    
    # --- init selection method ---
    q_prompt_format, s_prompt_format = args.prompt_format.split("-")[0], args.prompt_format.split("-")[1]
    selection_method = init_selection_method(
        args.selection, env, args.encoder, args.task, args.cand_split, args.seed,
        q_prompt_format=q_prompt_format, metric=args.metric, device=device,
        use_pi=args.use_pi, 
        # additional info for each selection method
        skill_encoder_path=args.skill_encoder,
        rewrite_cand_path=args.rewrite_cand_path,
        prompt_pg_ckpt=args.prompt_pg_ckpt,
        repeat_selection_path=args.repeat_selection_path,
    )
    
    # --- config logging ---
    encoder_type = args.encoder.replace("/", "|")
    extra = ""
    if args.skill_encoder is not None:
        extra = os.path.basename(os.path.dirname(args.skill_encoder)) + "/"

    if args.rewrite_cand_path is not None:
        extra = os.path.basename(os.path.dirname(args.rewrite_cand_path)) + "/"

    if args.prompt_pg_ckpt is not None:
        extra = os.path.basename(os.path.dirname(args.prompt_pg_ckpt)) + "/"

    result_path = f"results/eval.py/{args.task}/{args.split}/{args.selection}/{extra}{args.endpoint}/{encoder_type}/{args.cand_split}-shot_{args.shot_number}-prompt_{args.prompt_format}"
    os.makedirs(result_path, exist_ok=True)

    result_file = "{}/seed_{}.json".format(result_path, args.seed)
    # load the check point
    if os.path.exists(result_file):
        print(f"# The result file {result_file} exists! We will load the check point!!!")
        check_point = json.load(open(result_file))
        results = check_point['results']
        test_uids = [uid for uid in test_uids if uid not in results.keys()]
    else:
        results = {}

    # --- run evaluation ---
    while len(test_uids) > 0:
        uid = "DUMMY_UID"
        # reset until uid in the test set
        while uid not in test_uids:
            obs, info = env.reset()
            id = info['test_id']
            uid = info['test_uid']
            done = False
            print("Loading " + str(uid))

        while not done:
            action = selection_method.select(id)
            obs, reward, done, _, info = env.step(action)
        print("Answer: %s" %info['answer'])
        test_uids.remove(uid)
        p = test_dataset.get_problem(id)
        results_ = {
            uid: {
                "answer": info["answer"],
                "prediction": info["prediction"],
                "correct": reward == 1,  #info["answer"] == info["prediction"],
                "shot_uids_steps": info["shot_uids_steps"],
                "outputs": "\\n".join(info["analyzer_gens"]),
            }
        }
        results.update(results_)

        if len(results) % args.save_every == 0:
            data = {}
            data['accuracy'] = float(corrects) / len(results)
            data['correct'] = int(corrects)
            data['count'] = int(len(results))
            data['configs'] = cfg
            data['results'] = results

            with open(result_file, 'w') as f:
                json.dump(data, f, indent=2, separators=(',', ': '))

        # print to the screen
        corrects = np.sum([r["correct"] for r in results.values()])
        print(f">>>>>>>>>> Progress    {len(results)} / {len(test_dataset)} <<<<<<<<<<")
        print(f">>>>>>>>>> Correctness {corrects} / {len(results)} <<<<<<<<<<")
        if args.verbose:
            print("--------- Question ---------")
            print(info['analyzer_prompts'][0].split('\n\n')[-1])
            print("--------- Generation ---------")
            print("\\n".join(info["analyzer_gens"]))
            print("--------- Verification ---------")
            print("Answer: %s; Prediction: %s" %(info['answer'], info["prediction"]))

        print("\n")
    
    corrects = np.sum([r["correct"] for r in results.values()])
    data = {}
    data['accuracy'] = float(corrects) / len(results)
    data['correct'] = int(corrects)
    data['count'] = int(len(results))
    data['configs'] = cfg
    data['results'] = results

    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))

