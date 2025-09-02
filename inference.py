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
from utils import resolve_file_name, ResolvedFilePath, organize_evaluation_results
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

        os.makedirs(res_saving_path, exist_ok=True)

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