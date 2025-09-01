# This file is used to conduct inference.
# to make the data generation more aligned, we use the same parquet file for inference.

from dataclasses import dataclass
from llm import vLLM
from utils import read_parquet, save_parquet, setup_logging, extract_answer_or_all, extract_evidence_or_none
from typing import List, Dict, Literal
from evaluator import Evaluator
import json, os
from tqdm import tqdm
from configs import LLMConfig, InferenceConfigs
from utils import resolve_file_name, ResolvedFilePath
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path", type=str, nargs="+")
parser.add_argument("--llm_name", type=str)
parser.add_argument("--url", type=str)
args = parser.parse_args()

paths = args.path
llm_name = args.llm_name
url = args.url

logger = setup_logging("Inference")
os.environ['TRANSFORMERS_CACHE'] = "/root/.cache/huggingface/hub/"



custom_InferenceConfigs = InferenceConfigs(
    llm_config = LLMConfig(
        model_name = llm_name,
        url = url
    ),
    data_path = paths
)

custom_llm = vLLM(custom_InferenceConfigs.llm_config)

def inference(data: List[Dict], resume: int = 0):
    for i, item in enumerate(tqdm(data[resume:], desc="Inference")):
        sys_prompt = item["prompt"][0]
        user_input = item["prompt"][1]
        answer = custom_llm.generate(user_input, sys_prompt)
        if answer == "ERROR: THE MODEL CANNOT PROCESS THE REQUEST.":
            logger.error(f"Fail to process the input, the user input is {user_input}, the sys prompt is {sys_prompt}")
            logger.error(f"please resume the inference process from number {resume + i}.")
            break
        item["answer"] = answer
    return data


if __name__ == "__main__":
    for data_path in custom_InferenceConfigs.data_path:
        data = read_parquet(data_path)

        llm_name = custom_InferenceConfigs.llm_config.model_name.split("/")[-1]

        resolved_file_name = resolve_file_name(data_path)
        split = resolved_file_name.split
        Dataset_name = resolved_file_name.dataset
        data_chunk_size = resolved_file_name.chunk_size
        retriever = resolved_file_name.retriever
        reranker = resolved_file_name.reranker
        top_k = resolved_file_name.top_k
        wogt_rate = resolved_file_name.wogt_rate
        generate_method = resolved_file_name.method

        res_saving_path = f"outputs/{Dataset_name}/{llm_name}/{split}/{data_chunk_size}"
        os.makedirs(res_saving_path, exist_ok=True)
        saving_path = os.path.join(res_saving_path, data_path.replace(".parquet", f"_{llm_name}_inference.parquet").split("/")[-1])
        logger.info(f"inference {data_path} with {llm_name} now.")

        data = inference(data)
        save_parquet(data, saving_path)

        logger.info(f"inference done, the data is saved to {saving_path}")
        logger.info(f"the first data is: {data[0]}")

        os.makedirs(res_saving_path, exist_ok=True)

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
        with open(os.path.join(res_saving_path, f"{generate_method}_{retriever}_{reranker}_{top_k}_{wogt_rate}_all_results.json"), "w") as f:
            json.dump(all_results, f, indent=4)