from argparse import ArgumentParser
import pandas as pd
import random
from typing import Literal
from collections import Counter


def deduplicate_results(results):
    seen = set()
    unique_results_idx = []

    for i, r in enumerate(results):
        key = (
            frozenset(r)
        )
        if key not in seen:
            seen.add(key)
            unique_results_idx.append(i)

    return unique_results_idx

parser = ArgumentParser()
parser.add_argument("--path", type=str,nargs='+', default=[])
parser.add_argument("--output_path", type=str, default="data/merged/0822-QASPER.parquet")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--deduplicate", type=bool, default=False)
parser.add_argument("--deduplicate_type", type=str, default="hard", choices=["hard", "soft"])
args = parser.parse_args()

paths = args.path
output_path = args.output_path
seed = args.seed
deduplicate = args.deduplicate
deduplicate_type = args.deduplicate_type

all_data = []

for path in paths:
    data = pd.read_parquet(path)
    data = data.to_dict(orient="records")
    if not deduplicate:
        all_data.extend(data)
    else:
        all_data.append(data)

if deduplicate and deduplicate_type == "hard":
    all_data = [list(x) for x in zip(*all_data)]
    print(len(all_data))
    filtered_cnt = 0
    filtered_data = []
    for data in all_data:
        # data is a list of dicts.
        retrieved_text_set = set()
        for entry in data:
            if entry["prompt"][1]["content"] in retrieved_text_set:
                filtered_cnt += 1
                continue
            else:
                retrieved_text_set.add(entry["prompt"][1]["content"])
                filtered_data.append(entry)
    print(len(filtered_data))
    print(f"filtered {filtered_cnt} entries.")
    all_data = filtered_data

if deduplicate and deduplicate_type == "soft":
    all_data = [list(x) for x in zip(*all_data)]
    print(len(all_data))
    print(len(all_data[0]))
    filtered_cnt = 0
    filtered_data = []
    for data in all_data:
        seen = set()
        unique_lists = []
        for entry in data:
            retrieved = entry["prompt"][1]["content"].split("<ref>")[1].split("</ref>")[0].strip()
            retrieved_chunks = [text.partition(".")[2].strip() for text in retrieved.split("\n")]
            unique_lists.append(retrieved_chunks)
        unique_idx = deduplicate_results(unique_lists)
        filtered_data.extend([data[i] for i in unique_idx])
    print(len(filtered_data))
    all_data = filtered_data

random.seed(seed)
random.shuffle(all_data)

df = pd.DataFrame(all_data)
df.to_parquet(output_path)