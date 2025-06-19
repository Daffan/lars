import re

def answer_cleansing(dataset, pred):
    # print("dataset : " + dataset)
    # print("pred_before : " + pred)
    
    # if method in ("few_shot", "few_shot_cot"):
    direct_answer_trigger_for_fewshot = "The answer is"
    preds = pred.split(direct_answer_trigger_for_fewshot)
    answer_flag = True if len(preds) > 1 else False 
    before_pred = preds[-1]

    if dataset in ("aqua", "commonsensqa"):
        pred = re.findall(r'A|B|C|D|E', before_pred)
    elif dataset in ("bigbench_date", "date_understanding"):
        pred = re.findall(r'A|B|C|D|E|F', before_pred)
    elif dataset in ("object_tracking", "shuffled_objects"):
        pred = re.findall(r'A|B|C', before_pred)
    elif dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        pred = before_pred.replace(",", "")
        pred = [str(round(float(s), 2)) for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif dataset in ("strategyqa", "coin_flip"):
        pred = before_pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no", "true", "false")]
    elif dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s","", before_pred)
        pred = [pred]
    else:
        raise ValueError(f"dataset {dataset} is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last element in list ...
            pred = pred[-1]
    
    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
    
    # print("pred_after : " + pred)
    
    return pred