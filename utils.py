import json
import os
from typing import List, Dict, Union
import pandas as pd
import logging
import re
import sys

def extract_evidence_or_none(text:str):
    # if the text contains <evidence>...</evidence> tag, return the evidence.
    # else, return the text.
    match = re.search(r"<evidence>(.*?)</evidence>", text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return ""

def extract_answer_or_all(text:str):
    # if the text contains <answer>...</answer> tag, return the answer.
    # else, return the text.
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return text

# Configure logging
def setup_logging(name, level=logging.INFO, log_file=None):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    logger.propagate = False 

    return logger


def load_raw_qasper_data(data_folder: str, split: Union[List[str], str] = "train"):
    all_paper_data = {}
    all_QA_data = {}
    if isinstance(split, str):
        split = [split]
    for s in split:
        path = os.path.join(data_folder, s)
        paper_path = os.path.join(path, "paper_data.json")
        QA_path = os.path.join(path, "QA_data.json")
        with open(paper_path, "r") as f:
            paper_data = json.load(f)
        with open(QA_path, "r") as f:
            QA_data = json.load(f)
        all_paper_data[s] = paper_data
        all_QA_data[s] = QA_data
    return all_paper_data, all_QA_data

def save_parquet(data: List[Dict], path: str):
    df = pd.DataFrame(data)
    df.to_parquet(path)

def merge_by_reverse_removal(a: list[int], b: list[int]) -> list[int]:
    """
    a : the retrieved evidence chunk ids.
    b : the gt_ground truth chunk ids to merge into a.
    return : the merged chunk ids, the length should be equal to max(len(a), len(b)).
    """
    set_a = set(a)
    set_b = set(b)

    to_add = [x for x in b if x not in set_a]  # O(len(b))
    to_remove = set_a - set_b  # O(len(a))

    num_to_remove = len(to_add)
    result = []
    removed = 0

    for x in reversed(a):
        if x in to_remove and removed < num_to_remove:
            removed += 1
        else:
            result.append(x)

    result.reverse()
    result.extend(to_add)

    return result

def read_parquet(path: str) -> List[Dict]:
    return pd.read_parquet(path).to_dict(orient="records")

def safe_len(x):
    if x is None:
        return 1  # None 表示单配置，等价于长度1
    return len(x)

def organize_evaluation_results(all_results: Dict[str, Dict[str, Dict[str, List[float]]]]) -> dict:
    organized_results = {}
    for group_name, group_results in all_results.items():
        organized_results[group_name] = {
            "count": group_results["count"],
            }
        for metrics_type, value in group_results.items():
            if metrics_type != "count":
                organized_results[group_name][metrics_type] = {}
                for metric, metric_results in value.items():
                    if metric != "choice":
                        organized_results[group_name][metrics_type][metric] = sum(metric_results) / len(metric_results)
                    else:
                        evidence_rate = [metric_result == "<evidence>" for metric_result in metric_results].count(True) / len(metric_results)
                        organized_results[group_name][metrics_type]["choice"] = evidence_rate

    return organized_results
    
if __name__ == "__main__":
    b = [4, 17, 52, 6]
    a = [3,4,17,73,5,8,10,6]
    print(merge_by_reverse_removal(a, b))