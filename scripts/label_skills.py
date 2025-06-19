import os
import json
import numpy as np

tabmwp_idx2label = {
    0: "Compute statistics",
    1: "Compute rate of change",
    2: "Compute money cost",
    3: "Filter tree leaves",
    4: "Addtion/subtraction",
    5: "Search minimum/maximum",
    6: "Multiplication",
    7: "Filter table entries",
    8: "Compute probability",
    9: "Shortage or surplus?",
    10: "Reason time schedule",
    11: "Compare numbers",
    12: "Others",
}

# Consider sparate money cost: addition, subtraction, multiplication
# But some instances may contains more than one

def label_TabMWP():
    np.random.seed(11)
    data_root = "assets/dataset/tablemwp"
    split = "problems_train_cand1k_labeled.json"

    # Load data
    data = json.load(open(os.path.join(data_root, split)))
    data_load = json.load(open(os.path.join(data_root, "problems_train_cand1k_labeled_new.json")))
    data_with_label = {}

    for i, k in enumerate(data):
        # print the dict of available labels
        print("Available labels: ")
        for idx, label in tabmwp_idx2label.items():
            print("{}: {}".format(idx, label))
        print("Progress {}/{}".format(i, len(data)))
        print("Question: ", data[k]["question"])
        print("solution: ", data[k]["solution"])

        # read keyboard input
        data_with_label[k] = data[k]
        if k in data_load:
            label = data_load[k]["label"]
            data_with_label[k]["label"] = label
        else:
            label = input(f"Old label: {data[k]['label']}; new label: ")
            data_with_label[k]["label"] = label if label != "" else data[k]["label"]

        # clear the page
        os.system('clear')

        if (i + 1) % 10 == 0:
            # Save data
            json.dump(data_with_label, open(os.path.join(data_root, "problems_train_cand1k_labeled_1116.json"), "w"), indent=2, separators=(',', ': '))

    # Save data
    json.dump(data_with_label, open(os.path.join(data_root, "problems_train_cand1k_labeled_1116.json"), "w"), indent=2, separators=(',', ': '))

if __name__ == "__main__":
    label_TabMWP()