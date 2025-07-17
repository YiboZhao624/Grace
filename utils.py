import json
import os
from typing import List, Dict
import pandas as pd

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