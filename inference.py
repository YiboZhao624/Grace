# This file is used to conduct inference.
# to make the data generation more aligned, we use the same parquet file for inference.

from llm import vLLM
from utils import read_parquet, save_parquet, setup_logging, extract_answer_or_all, extract_evidence_or_all
from configs import InferenceConfigs
from typing import List, Dict, Literal
from evaluator import Evaluator
import json


logger = setup_logging("Inference")

custom_InferenceConfigs = InferenceConfigs(
    data_path = ["data/processed/0806-256cs/QASPER_Split-test_Prompt-default_NumberTemplate-False_retriever_bm25_TopK-7_WO_GT_Evidence_Rate-0.2.parquet","data/processed/0806-256cs/QASPER_Split-test_Prompt-default_NumberTemplate-False_retriever_reranker_vllm-Qwen3-Embedding-0.6B_TopK-7_WO_GT_Evidence_Rate-0.2.parquet","data/processed/0806-256cs/QASPER_Split-test_Prompt-default_NumberTemplate-False_retriever_vllm-Qwen3-Embedding-0.6B_TopK-7_WO_GT_Evidence_Rate-0.2.parquet"]
)
custom_llm = vLLM(custom_InferenceConfigs.llm_config)

def inference(data: List[Dict]):
    for item in data:
        input_prompt = item["prompt"]
        answer = custom_llm.generate(input_prompt)
        item["answer"] = answer
    return data


if __name__ == "__main__":
    data = read_parquet(custom_InferenceConfigs.data_path)
    inference(data)
    save_parquet(data, custom_InferenceConfigs.data_path)
    logger.info(f"inference done, the data is saved to {custom_InferenceConfigs.data_path}")
    logger.info(f"the first data is: {data[0]}")
    
    enabled_metrics = ["BS","EM","RL","BL","RR"]
    kwargs = {
        "BERT_path": "bert-base-uncased",
        "device": "cuda:0"
    }
    evaluator = Evaluator(metrics=enabled_metrics, **kwargs)
    full_answers = [item["answer"] for item in data]
    chosen_evidences = [extract_evidence_or_all(item["answer"]) for item in data]
    answers = [extract_answer_or_all(item["answer"]) for item in data]
    ground_truths = [item["ground_truth"] for item in data]
    ground_truth_evidences = [item["ground_truth_evidences"] for item in data]
    answer_results, evidence_results, reward_results = evaluator.evaluate(full_answers, chosen_evidences, answers, ground_truths, ground_truth_evidences)

    # aggregate the results in each dict.
    for key, value in answer_results.items():
        answer_results[key] = sum(value) / len(value)
    for key, value in evidence_results.items():
        evidence_results[key] = sum(value) / len(value)
    for key, value in reward_results.items():
        reward_results[key] = sum(value) / len(value)
    
    with open("answer_results.json", "w") as f:
        json.dump(answer_results, f, indent=4)
    with open("evidence_results.json", "w") as f:
        json.dump(evidence_results, f, indent=4)