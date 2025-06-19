import os
import json
import argparse
from collections import defaultdict

import numpy as np

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def split_tablemwp(seed=None):
    np.random.seed(11)
    data_root = "assets/dataset/tablemwp"

    # split TableMWP data
    # split the dev split into dev_remain dev_cand1k dev_cand100
    dev = json.load(open(os.path.join(data_root, f'problems_dev.json')))
    pids = list(dev.keys())
    np.random.shuffle(pids)
    
    dev_cand100_pids = pids[:100]
    dev_cand1k_pids = pids[:1000]
    dev_remain_pids = pids[1000:]

    dev_cand100 = {pid: dev[pid] for pid in dev_cand100_pids}
    dev_cand1k = {pid: dev[pid] for pid in dev_cand1k_pids}
    dev_remain = {pid: dev[pid] for pid in dev_remain_pids}

    json.dump(dev_cand100, open(os.path.join(data_root, f'problems_dev_cand100.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(dev_cand1k, open(os.path.join(data_root, f'problems_dev_cand1k.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(dev_remain, open(os.path.join(data_root, f'problems_dev_remain.json'), 'w'), indent=2, separators=(',', ': '))

    # split the train split into train_remain train_cand1k train_cand100
    train = json.load(open(os.path.join(data_root, f'problems_train.json')))
    pids = list(train.keys())
    np.random.shuffle(pids)

    train_cand100_pids = pids[:100]
    train_cand200_pids = pids[:200]
    train_cand1k_pids = pids[:1000]
    train_remain_pids = pids[1000:]

    train_cand100 = {pid: train[pid] for pid in train_cand100_pids}
    train_cand200 = {pid: train[pid] for pid in train_cand200_pids}
    train_cand1k = {pid: train[pid] for pid in train_cand1k_pids}
    train_remain = {pid: train[pid] for pid in train_remain_pids}

    json.dump(train_cand100, open(os.path.join(data_root, f'problems_train_cand100.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(train_cand200, open(os.path.join(data_root, f'problems_train_cand200.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(train_cand1k, open(os.path.join(data_root, f'problems_train_cand1k.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(train_remain, open(os.path.join(data_root, f'problems_train_remain.json'), 'w'), indent=2, separators=(',', ': '))

def split_gsm8k(seed=None):
    np.random.seed(11)
    data_root = "assets/dataset/grade-school-math/grade_school_math/data"

    # split GSM8K data
    # split the train split into train_remain train_cand1k train_cand100
    train = read_jsonl(os.path.join(data_root, f'train.jsonl'))
    pids = list(range(len(train)))
    np.random.shuffle(pids)

    train_all_pids = pids[:]
    train_cand100_pids = pids[:100]
    train_cand1k_pids = pids[:1000]
    train_remain_pids = pids[1000:]

    train_all = dict(zip([
        "train" + str(pid) for pid in train_all_pids],
        [train[pid] for pid in train_all_pids]
    ))
    train_cand100 = dict(zip([
        "train" + str(pid) for pid in train_cand100_pids],
        [train[pid] for pid in train_cand100_pids]
    ))
    train_cand1k = dict(zip([
        "train" + str(pid) for pid in train_cand1k_pids],
        [train[pid] for pid in train_cand1k_pids]
    ))
    train_remain = dict(zip([
        "train" + str(pid) for pid in train_remain_pids],
        [train[pid] for pid in train_remain_pids]
    ))

    json.dump(train_all, open(os.path.join(data_root, f'train_all.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(train_cand100, open(os.path.join(data_root, f'train_cand100.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(train_cand1k, open(os.path.join(data_root, f'train_cand1k.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(train_remain, open(os.path.join(data_root, f'train_remain.json'), 'w'), indent=2, separators=(',', ': '))
    print(f"train split done with {len(train)} samples")

    # reformat the test split
    test = read_jsonl(os.path.join(data_root, f'test.jsonl'))
    test_reformat = dict(zip([
        "test" + str(pid) for pid in range(len(test))],
        test
    ))

    json.dump(test_reformat, open(os.path.join(data_root, f'test_reformat.json'), 'w'), indent=2, separators=(',', ': '))
    print(f"test split done with {len(test)} samples")

def split_aqua(seed=None):
    np.random.seed(11)
    data_root = "assets/dataset/AQuA"

    # split AQuA data
    # split the dev split into dev_remain dev_cand1k dev_cand100
    with open(os.path.join(data_root, "dev.json"), 'r') as handle:
        dev = [json.loads(line) for line in handle]
    pids = list(range(len(dev)))
    np.random.shuffle(pids)
    dev_cand100_pids = pids[:100]
    dev_cand1k_pids = pids[:1000]
    dev_remain_pids = pids[1000:]

    dev_cand100 = dict(zip([
        "dev" + str(pid) for pid in dev_cand100_pids],
        [dev[pid] for pid in dev_cand100_pids]
    ))
    dev_cand1k = dict(zip([
        "dev" + str(pid) for pid in dev_cand1k_pids],
        [dev[pid] for pid in dev_cand1k_pids]
    ))
    dev_remain = dict(zip([
        "dev" + str(pid) for pid in dev_remain_pids],
        [dev[pid] for pid in dev_remain_pids]
    ))

    json.dump(dev_cand100, open(os.path.join(data_root, f'dev_cand100.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(dev_cand1k, open(os.path.join(data_root, f'dev_cand1k.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(dev_remain, open(os.path.join(data_root, f'dev_remain.json'), 'w'), indent=2, separators=(',', ': '))
    print(f"dev split done with {len(dev)} samples")

    # split the train split into train_remain train_cand1k train_cand100
    with open(os.path.join(data_root, "train.json"), 'r') as handle:
        train = [json.loads(line) for line in handle]
    pids = list(range(len(train)))
    np.random.shuffle(pids)

    train_cand100_pids = pids[:100]
    train_cand1k_pids = pids[:1000]
    train_10k_pids = pids[1000:11000]
    train_remain_pids = pids[11000:]

    train_cand100 = dict(zip([
        "train" + str(pid) for pid in train_cand100_pids],
        [train[pid] for pid in train_cand100_pids]
    ))
    train_cand1k = dict(zip([
        "train" + str(pid) for pid in train_cand1k_pids],
        [train[pid] for pid in train_cand1k_pids]
    ))
    train_10k = dict(zip([
        "train" + str(pid) for pid in train_10k_pids],
        [train[pid] for pid in train_10k_pids]
    ))
    train_remain = dict(zip([
        "train" + str(pid) for pid in train_remain_pids],
        [train[pid] for pid in train_remain_pids]
    ))

    json.dump(train_cand100, open(os.path.join(data_root, f'train_cand100.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(train_cand1k, open(os.path.join(data_root, f'train_cand1k.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(train_10k, open(os.path.join(data_root, f'train_10k.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(train_remain, open(os.path.join(data_root, f'train_remain.json'), 'w'), indent=2, separators=(',', ': '))
    print(f"train split done with {len(train)} samples")

    # load and save test
    with open(os.path.join(data_root, "test.json"), 'r') as handle:
        test = [json.loads(line) for line in handle]

    test_reformat = dict(zip([
        "test" + str(pid) for pid in range(len(test))],
        test
    ))

    json.dump(test_reformat, open(os.path.join(data_root, f'test_reformat.json'), 'w'), indent=2, separators=(',', ': '))
    print(f"test split done with {len(test)} samples")

def split_zero_shot_reasoner():
    np.random.seed(11)
    data_root = "assets/dataset/zero_shot_cot/log"
    dataset_train = {}
    dataset_train_correct = {}
    dataset_cand = {}
    dataset_cand_correct = {}
    dataset_test = {}
    correct_count_total = 0

    for log_file in os.listdir(data_root):
        if log_file.endswith("cot.log"):
            task_name = log_file.split("_zero_shot")[0]
            dataset_task = {}
            with open(os.path.join(data_root, log_file), 'r', encoding="utf-8") as fp:
                count = 0
                correct_count = 0
                answer_seg = ""
                for line in fp:
                    if "Q: " in line:
                        c_question = line.strip()
                    if "A: " in line:
                        answer_seg = line
                    elif "Therefore" in line and "the answer" in line:
                        c_rationale = answer_seg

                    elif answer_seg != "":
                        answer_seg += line
                    if "pred_mode" in line:
                        c_pred_ans = line.split(":")[1].strip()
                    if "GT :" in line:
                        c_gold_ans = line.split(":")[1].strip()

                        c_rationale = c_rationale.replace("A: Let's think step by step.", "").strip()
                        steps = [s.strip() for s in c_rationale.split("\n") if len(s.strip()) > 0]
                        c_rationale = "\n".join(steps)
                        c_question = c_question.replace("Q:", "").strip()
                        uid = f"{task_name}_{count}"
                        dataset_task.update({
                            uid: {
                                "task": task_name,
                                "question": c_question,
                                "rationale": c_rationale,
                                "correct": c_gold_ans,
                                "pred_ans": c_pred_ans
                            }
                        })
                        count += 1
                        correct_count += 1 if c_gold_ans == c_pred_ans else 0
            keys = list(dataset_task.keys())
            np.random.shuffle(keys)
            dataset_cand.update(dict((k, dataset_task[k]) for k in keys[:100]))
            dataset_cand_correct.update(dict((k, dataset_task[k]) for k in keys[:100] if dataset_task[k]["correct"] == dataset_task[k]["pred_ans"]))
            dataset_test.update(dict((k, dataset_task[k]) for k in keys[100:200]))
            dataset_train.update(dict((k, dataset_task[k]) for k in keys[200:]))
            dataset_train_correct.update(dict((k, dataset_task[k]) for k in keys[200:] if dataset_task[k]["correct"] == dataset_task[k]["pred_ans"]))
            print("Task: %s; Count: %d; Accuracy: %.2f" % (task_name, count, correct_count / count * 100))
            correct_count_total += correct_count
        
    # save the dataset
    print("Saving the dataset ...")
    print("Accuracy total: %.2f" % (correct_count_total / (len(dataset_train) + len(dataset_cand) + len(dataset_test)) * 100))
    print("Train: %d; Test: %d; Cand: %d" % (len(dataset_train), len(dataset_test), len(dataset_cand)))
    print("Train correct: %d; Cand correct: %d" % (len(dataset_train_correct), len(dataset_cand_correct)))
    json.dump(dataset_train, open(os.path.join(data_root, "..", f'train_remain.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(dataset_train_correct, open(os.path.join(data_root, "..", f'train_remain_correct.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(dataset_test, open(os.path.join(data_root, "..", f'test_reformat.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(dataset_cand, open(os.path.join(data_root, "..", f'train_cand1k.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(dataset_cand_correct, open(os.path.join(data_root, "..", f'train_cand1k_correct.json'), 'w'), indent=2, separators=(',', ': '))


def split_COGS(categ=["Primitive Substitution", "Primitive Structural Alternation"], all=False):
    np.random.seed(11)
    data_root = "assets/dataset/COGS/cofe"
    file_names = [f for f in os.listdir(data_root) if f == "full_similarity.json"]

    dataset_o = []
    for f in file_names:
        with open(os.path.join(data_root, f), 'r') as handle:
            sub_dataset = [json.loads(line) for line in handle]
            if not all:
                sub_dataset_c = [d for d in sub_dataset if d["category"] in categ]
            else:
                sub_dataset_c = sub_dataset
            dataset_o.extend(sub_dataset_c)

    dataset = {}
    for d in dataset_o:
        context = d["context"]
        input_output = [s for s in context.split("\n\n") if len(s.strip()) > 0]
        for io in input_output:
            if "output:" in io:
                i, o = io.split("\n")
                i = i.replace("input:", "").strip()
                o = o.replace("output:", "").strip()

            dataset[i] = {
                "question": i,
                "answer": o,
                "category": d["category"],
            }

    keys = list(range(len(dataset)))

    np.random.shuffle(keys)
    o_keys = list(dataset.keys())
    dataset_test = dict((k, dataset[o_keys[k]]) for k in keys[:1000])  # 1000 test samples
    dataset_train = dict((k, dataset[o_keys[k]]) for k in keys[1000:])  # 1000 test samples
    dataset_train_cand100 = dict((k, dataset[o_keys[k]]) for k in keys[1000:1100])  # 100 cand samples
    dataset_train_cand1k = dict((k, dataset[o_keys[k]]) for k in keys[1000:2000])  # 1000 cand samples

    print("Saving the dataset ...")
    print("Train: %d; Test: %d; Cand100: %d; Cand1k: %d" % (len(dataset_train), len(dataset_test), len(dataset_train_cand100), len(dataset_train_cand1k)))
    json.dump(dataset_train, open(os.path.join(data_root, "..", f'train_all.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(dataset_test, open(os.path.join(data_root, "..", f'test_reformat.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(dataset_train_cand100, open(os.path.join(data_root, "..", f'train_cand100.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(dataset_train_cand1k, open(os.path.join(data_root, "..", f'train_cand1k.json'), 'w'), indent=2, separators=(',', ': '))
    print("Done.")

def split_spider():
    np.random.seed(11)
    data_root = "assets/dataset/spider"

    databases = json.load(open(os.path.join(data_root, f'tables.json')))
    db_id_to_db = {}
    for db in databases:
        db_id_to_db[db['db_id']] = db

    def database_to_schema(db_id):
        db = db_id_to_db[db_id]
        table_to_columns_names = defaultdict(list)
        table_names = db['table_names_original']
        for entry in db['column_names_original']:
            if entry[0] >= 0:
                table_name = table_names[entry[0]]
                table_to_columns_names[table_name].append(entry[1])

        table_texts = [f"{table_name} ({', '.join(column_names)})" for table_name, column_names in table_to_columns_names.items()]
        return "\n".join(table_texts)

    # split Spider data
    # split the train split into train_remain train_cand1k train_cand100
    train = json.load(open(os.path.join(data_root, f'train_spider.json')))
    train_reformat = {}
    for pid, problem in enumerate(train):
        train_reformat[pid] = {
            "question": problem["question"],
            "schema": database_to_schema(problem["db_id"]),
            "answer": problem["query"],
            "db_id": problem["db_id"]
        }
    pids = list(range(len(train)))
    np.random.shuffle(pids)
    
    train_cand100_pids = pids[:100]
    train_cand1k_pids = pids[:1000]
    train_remain_pids = pids[1000:]
    train_all_pids = pids

    train_cand100 = {"train" + str(pid): train_reformat[pid] for pid in train_cand100_pids}
    train_cand1k = {"train" + str(pid): train_reformat[pid] for pid in train_cand1k_pids}
    train_remain = {"train" + str(pid): train_reformat[pid] for pid in train_remain_pids}
    train_all = {"train" + str(pid): train_reformat[pid] for pid in train_all_pids}

    # reformat test
    test = json.load(open(os.path.join(data_root, f'dev.json')))
    test_reformat = {}
    for pid, problem in enumerate(test):
        test_reformat["test" + str(pid)] = {
            "question": problem["question"],
            "schema": database_to_schema(problem["db_id"]),
            "answer": problem["query"],
            "db_id": problem["db_id"]
        }

    json.dump(train_cand100, open(os.path.join(data_root, f'train_cand100.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(train_cand1k, open(os.path.join(data_root, f'train_cand1k.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(train_remain, open(os.path.join(data_root, f'train_remain.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(train_all, open(os.path.join(data_root, f'train_all.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(test_reformat, open(os.path.join(data_root, f'test_reformat.json'), 'w'), indent=2, separators=(',', ': '))

    print(f"test split done with {len(test_reformat)} samples")
    print(f"train split done with {len(train_all_pids)} samples")  

try:
    from cot import Collection
except:
    pass

def create_split_from_ThoughtSource(dataset_name, num_test=1000):
    np.random.seed(11)
    dataset = Collection([dataset_name], verbose=False)
    
    pids = list(range(len(dataset.all_train)))
    np.random.shuffle(pids)

    test_pids = pids[:num_test]
    train_cand100_pids = pids[num_test:100]
    train_cand1k_pids = pids[num_test:1000]
    train_remain_pids = pids[num_test+1000:]
    train_all_pids = pids[num_test:]

    train_cand100 = {dataset.all_train[pid]["id"]: dataset.all_train[pid] for pid in train_cand100_pids}
    train_cand1k = {dataset.all_train[pid]["id"]: dataset.all_train[pid] for pid in train_cand1k_pids}
    train_remain = {dataset.all_train[pid]["id"]: dataset.all_train[pid] for pid in train_remain_pids}
    train_all = {dataset.all_train[pid]["id"]: dataset.all_train[pid] for pid in train_all_pids}

    # reformat test
    test_reformat = {dataset.all_train[pid]["id"]: dataset.all_train[pid] for pid in test_pids}

    if not os.path.exists(os.path.join("assets/dataset", "ThoughtSource_" + dataset_name)):
        os.makedirs(os.path.join("assets/dataset", "ThoughtSource_" + dataset_name))

    json.dump(train_cand100, open(os.path.join("assets/dataset", "ThoughtSource_" + dataset_name, f'train_cand100.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(train_cand1k, open(os.path.join("assets/dataset", "ThoughtSource_" + dataset_name, f'train_cand1k.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(train_remain, open(os.path.join("assets/dataset", "ThoughtSource_" + dataset_name, f'train_remain.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(train_all, open(os.path.join("assets/dataset", "ThoughtSource_" + dataset_name, f'train_all.json'), 'w'), indent=2, separators=(',', ': '))
    json.dump(test_reformat, open(os.path.join("assets/dataset", "ThoughtSource_" + dataset_name, f'test_reformat.json'), 'w'), indent=2, separators=(',', ': '))
    print(f"test split done with {len(test_pids)} samples")
    print(f"train split done with {len(train_all_pids)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create dataset splits for various datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/create_split.py --dataset tablemwp
  python scripts/create_split.py --dataset gsm8k
  python scripts/create_split.py --dataset commonsense_qa --num_test 500
  python scripts/create_split.py --dataset cogs --cogs_all
        """
    )
    parser.add_argument('--dataset', type=str, default='tablemwp', 
                       choices=['tablemwp', 'aqua', 'gsm8k', 'zero_shot_reasoner', 'spider', 
                               'commonsense_qa', 'strategy_qa', 'mawps', 'cogs'],
                       help='Dataset to process (default: tablemwp)')
    parser.add_argument('--num_test', type=int, default=1000,
                       help='Number of test samples for ThoughtSource datasets (default: 1000)')
    parser.add_argument('--cogs_all', action='store_true',
                       help='Process all COGS categories instead of just primitive ones')
    
    args = parser.parse_args()
    
    # Dataset processing function mapping
    dataset_functions = {
        'tablemwp': lambda: split_tablemwp(),
        'aqua': lambda: split_aqua(),
        'gsm8k': lambda: split_gsm8k(),
        'zero_shot_reasoner': lambda: split_zero_shot_reasoner(),
        'spider': lambda: split_spider(),
        'commonsense_qa': lambda: create_split_from_ThoughtSource("commonsense_qa", num_test=args.num_test),
        'strategy_qa': lambda: create_split_from_ThoughtSource("strategy_qa", num_test=args.num_test),
        'mawps': lambda: create_split_from_ThoughtSource("mawps", num_test=args.num_test),
        'cogs': lambda: split_COGS(all=args.cogs_all)
    }
    
    try:
        print(f"Processing dataset: {args.dataset}")
        dataset_functions[args.dataset]()
        print(f"Successfully processed {args.dataset} dataset")
    except KeyError:
        print(f"Error: Unknown dataset '{args.dataset}'")
        parser.print_help()
        exit(1)
    except Exception as e:
        print(f"Error processing dataset {args.dataset}: {str(e)}")
        exit(1)