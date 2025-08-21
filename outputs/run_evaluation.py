import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_parquet, save_parquet, setup_logging, extract_answer_or_all, extract_evidence_or_none
from evaluator import Evaluator
import json
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path", type=str, default="outputs/QASPER/Qwen3-8B/test/256cs/QASPER_Split-test_Prompt-default_NumberTemplate-False_retriever_reranker_vllm-Qwen3-Embedding-0.6B_TopK-7_WO_GT_Evidence_Rate-0.2_Qwen3-8B_inference.parquet")
args = parser.parse_args()

path = args.path

logger = setup_logging("Evaluator")

os.environ['TRANSFORMERS_CACHE'] = "/root/.cache/huggingface/hub/"

path = "outputs/QASPER/Qwen3-8B/test/256cs/QASPER_Split-test_Prompt-default_NumberTemplate-False_retriever_reranker_vllm-Qwen3-Embedding-0.6B_TopK-7_WO_GT_Evidence_Rate-0.2_Qwen3-8B_inference.parquet"

res_saving_path = "/".join(path.split("/")[:-1])
generate_method = path.split("/")[-1].split("_WO_GT_Evidence")[0].split("_")[-3:-1]
generate_method = "_".join(generate_method)
W_GT_Evidence = path.split("/")[-1].split("_WO_GT_Evidence")[1].split("_")[0]

os.makedirs(res_saving_path, exist_ok=True)

data = read_parquet(path)

# logger.info(f"the first data is: {data[0]}")



full_answers = [item["answer"] for item in data]
chosen_evidences = [extract_evidence_or_none(item["answer"]) for item in data]
answers = [extract_answer_or_all(item["answer"]) for item in data]

references = [item["extra_info"]["references"] for item in data]
# reference is a list of gt_answers, for a single question.
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
    reward_results[key] = sum(value) / len(value)

all_results = {
    "answer_results": answer_results,
    "evidence_results": evidence_results,
    "reward_results": reward_results,
}
with open(os.path.join(res_saving_path, f"{generate_method}_{W_GT_Evidence}_all_results.json"), "w") as f:
    json.dump(all_results, f, indent=4)

# with open(os.path.join(res_saving_path, f"{generate_method}_{W_GT_Evidence}_answer_results.json"), "w") as f:
#     json.dump(answer_results, f, indent=4)
# with open(os.path.join(res_saving_path, f"{generate_method}_{W_GT_Evidence}_evidence_results.json"), "w") as f:
#     json.dump(evidence_results, f, indent=4)
# with open(os.path.join(res_saving_path, f"{generate_method}_{W_GT_Evidence}_reward_results.json"), "w") as f:
#     json.dump(reward_results, f, indent=4)