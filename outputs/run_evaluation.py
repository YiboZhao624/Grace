import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from utils import read_parquet, save_parquet, setup_logging, extract_answer_or_all, extract_evidence_or_none
from evaluator import Evaluator
import json
from tqdm import tqdm
from argparse import ArgumentParser
from typing import List, Dict
from utils import organize_evaluation_results, setup_logging, resolve_file_name, ResolvedFilePath


parser = ArgumentParser()
parser.add_argument("--path", type=str)
args = parser.parse_args()

path = args.path

logger = setup_logging("Evaluator")

os.environ['TRANSFORMERS_CACHE'] = "/root/.cache/huggingface/hub/"

# saving the res at the same folder as the inference results.
res_saving_path = "/".join(path.split("/")[:-1])

resolved_file_name = resolve_file_name(path)
Dataset_name = resolved_file_name.dataset
data_chunk_size = resolved_file_name.chunk_size
retriever = resolved_file_name.retriever
reranker = resolved_file_name.reranker
top_k = resolved_file_name.top_k
wogt_rate = resolved_file_name.wogt_rate
generate_method = resolved_file_name.method

os.makedirs(res_saving_path, exist_ok=True)

data = read_parquet(path)

logger.info("Initializing the evaluator...")
enabled_metrics = ["BS","EM","RL","BL","RR"]
kwargs = {
    "BERT_path": "bert-base-uncased",
    "device": "cuda:0"
}
evaluator = Evaluator(metrics=enabled_metrics, **kwargs)
logger.info("Evaluator initialized.")

all_results = evaluator.evaluate(data)
organized_results = organize_evaluation_results(all_results)

with open(os.path.join(res_saving_path, f"{generate_method}_{wogt_rate}_{retriever}_{reranker}_{top_k}_{data_chunk_size}_all_results.json"), "w") as f:
    json.dump(organized_results, f, indent=4)