import json
import os
from typing import List, Dict
import pandas as pd
import logging
import re

def extract_evidence_or_all(text:str):
    # if the text contains <evidence>...</evidence> tag, return the evidence.
    # else, return the text.
    match = re.search(r"<evidence>(.*?)</evidence>", text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return text

def extract_answer_or_all(text:str):
    # if the text contains <answer>...</answer> tag, return the answer.
    # else, return the text.
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return text

# Configure logging
def setup_logging(level=logging.INFO, log_file=None):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(),  # Console handler
            *([logging.FileHandler(log_file)] if log_file else [])
        ],
    )
    return logging.getLogger(__name__)

# Create a default logger
logger = setup_logging()

def load_raw_qasper_data(data_folder: str, split = "train"):
    path = os.path.join(data_folder, split)
    paper_path = os.path.join(path, "paper_data.json")
    QA_path = os.path.join(path, "QA_data.json")
    with open(paper_path, "r") as f:
        paper_data = json.load(f)
    with open(QA_path, "r") as f:
        QA_data = json.load(f)
    return paper_data, QA_data

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

if __name__ == "__main__":
    b = [4, 17, 52, 6]
    a = [3,4,17,73,5,8,10,6]
    print(merge_by_reverse_removal(a, b))