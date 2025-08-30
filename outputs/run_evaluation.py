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

@dataclass
class ResolvedFilePath:
    dataset: str
    split: str
    method: str
    chunk_size: str
    retriever: str
    reranker: str
    top_k: str
    wogt_rate: str

def resolve_file_name(file_name: str) -> ResolvedFilePath:
    file_name_list = file_name.split("/")[-1].split("-")
    resolved_file_name = ResolvedFilePath(
        dataset = file_name_list[0],
        split = file_name_list[1],
        method = file_name_list[2],
        chunk_size = file_name_list[3],
        retriever = file_name_list[4],
        reranker = file_name_list[5],
        top_k = file_name_list[6],
        wogt_rate = file_name_list[7]
    )
    return resolved_file_name

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

full_answers = [item["answer"] for item in data]
chosen_evidences = [extract_evidence_or_none(item["answer"]) for item in data]
answers = [extract_answer_or_all(item["answer"]) for item in data]

references = [item["extra_info"]["references"] for item in data]
ground_truths = []
ground_truth_evidences = []
unanswerable_index = []
for i, reference in enumerate(references):
    answer_list = []
    evidence_list = []
    for gt_answer in reference:
        answer_list.append(gt_answer["answer"])
        evidence_list.extend(gt_answer["evidence"])
    
    if len(evidence_list) == 0:
        if reference[0]["answer"] == "Unanswerable":
            # logger.warning(f"No evidence found for reference: {reference} at index {i}")
            unanswerable_index.append(i)
            evidence_list = [""]

    ground_truths.append(answer_list)
    ground_truth_evidences.append(evidence_list)

unanswerable_index = list(set(unanswerable_index))
logger.info(f"the length of unanswerable_index is: {len(unanswerable_index)}")
# logger.info(f"the unanswerable_index is: {unanswerable_index}")

logger.info(f"the length of full_answers is: {len(full_answers)}")
logger.info(f"the length of chosen_evidences is: {len(chosen_evidences)}")
logger.info(f"the length of answers is: {len(answers)}")
logger.info(f"the length of ground_truths is: {len(ground_truths)}")
logger.info(f"the length of ground_truth_evidences is: {len(ground_truth_evidences)}")


logger.info("Initializing the evaluator...")
enabled_metrics = ["BS","EM","RL","BL","RR"]
kwargs = {
    "BERT_path": "bert-base-uncased",
    "device": "cuda:0"
}
evaluator = Evaluator(metrics=enabled_metrics, **kwargs)
logger.info("Evaluator initialized.")


answer_results, evidence_results, reward_results = evaluator.evaluate(full_answers, chosen_evidences, answers, ground_truths, ground_truth_evidences)

# aggregate the results in each dict.
for key, value in answer_results.items():
    answer_results[key] = sum(value) / len(value)
for key, value in evidence_results.items():
    evidence_results[key] = sum(value) / len(value)
for key, value in reward_results.items():
    try:
        reward_results[key] = sum(value) / len(value)
    except Exception as e:
        logger.error(f"Error aggregating {key}: {e}")
        reward_results[key] = 0


all_results = {
    "answer_results": answer_results,
    "evidence_results": evidence_results,
    "reward_results": reward_results,
}
with open(os.path.join(res_saving_path, f"{generate_method}_{wogt_rate}_{retriever}_{reranker}_{top_k}_{data_chunk_size}_all_results.json"), "w") as f:
    json.dump(all_results, f, indent=4)