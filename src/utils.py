import json
import os
from typing import List, Dict, Union, Optional, Tuple
from copy import deepcopy
import pandas as pd
import logging
import re
import sys
from dataclasses import dataclass
import yaml
import argparse

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
    file_name_list = file_name.split("/")[-1].replace(".parquet", "").split("-")
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


def _normalize_splits(split: Union[List[str], str]) -> List[str]:
    if isinstance(split, str):
        return [split]
    return list(split)


def _build_full_text_from_contexts(contexts: List[Dict[str, str]]) -> str:
    text_parts: List[str] = []
    for context in contexts or []:
        title = context.get("title", "") if isinstance(context, dict) else ""
        text = context.get("text", "") if isinstance(context, dict) else ""
        combined = "\n".join(part.strip() for part in [title, text] if part and part.strip())
        if combined:
            text_parts.append(combined)
    return "\n\n".join(text_parts).strip()


class BaseDatasetLoader:
    def __init__(self, data_folder: str, split: Union[List[str], str]):
        self.data_folder = data_folder
        self.splits = _normalize_splits(split)

    def load(self) -> Tuple[Dict[str, Dict], Dict[str, List[Dict]]]:
        raise NotImplementedError


class QasperDatasetLoader(BaseDatasetLoader):
    def load(self) -> Tuple[Dict[str, Dict], Dict[str, List[Dict]]]:
        all_paper_data: Dict[str, Dict[str, Dict]] = {}
        all_QA_data: Dict[str, List[Dict]] = {}
        for s in self.splits:
            path = os.path.join(self.data_folder, s)
            paper_path = os.path.join(path, "paper_data.json")
            QA_path = os.path.join(path, "QA_data.json")
            with open(paper_path, "r", encoding="utf-8") as f:
                paper_data = json.load(f)
            with open(QA_path, "r", encoding="utf-8") as f:
                QA_data = json.load(f)
            all_paper_data[s] = paper_data
            all_QA_data[s] = QA_data
        return all_paper_data, all_QA_data


class MetadataBackedDatasetLoader(BaseDatasetLoader):
    dataset_name: str = ""

    def load(self) -> Tuple[Dict[str, Dict], Dict[str, List[Dict]]]:
        all_paper_data: Dict[str, Dict[str, Dict]] = {}
        all_QA_data: Dict[str, List[Dict]] = {}
        for s in self.splits:
            qa_path = os.path.join(self.data_folder, s, "QA_data.json")
            with open(qa_path, "r", encoding="utf-8") as f:
                qa_records: List[Dict] = json.load(f)

            paper_data_split: Dict[str, Dict] = {}
            qa_list: List[Dict] = []
            for idx, record in enumerate(qa_records):
                qa_entry = deepcopy(record)
                paper_id = qa_entry.get("paper_id") or qa_entry.get("question_id") or f"{s}_{idx}"
                metadata = qa_entry.get("metadata", {}) or {}
                contexts = metadata.get("context") or []
                full_text = _build_full_text_from_contexts(contexts) or qa_entry.get("question", "")

                paper_data_split[paper_id] = {
                    "title": contexts[0].get("title", "") if contexts else metadata.get("title", ""),
                    "abstract": metadata.get("abstract", ""),
                    "full_text": full_text,
                    "contexts": contexts,
                }

                qa_entry["paper_id"] = paper_id
                metadata["context"] = contexts
                metadata["data_source"] = self.dataset_name or metadata.get("data_source", "")
                qa_entry["metadata"] = metadata
                qa_list.append(qa_entry)

            all_paper_data[s] = paper_data_split
            all_QA_data[s] = qa_list

        return all_paper_data, all_QA_data


class HotpotDatasetLoader(MetadataBackedDatasetLoader):
    dataset_name = "HotpotQA"


class TwoWikiDatasetLoader(MetadataBackedDatasetLoader):
    dataset_name = "2WikiMultiHopQA"


def load_raw_qasper_data(data_folder: str, split: Union[List[str], str] = "train"):
    return QasperDatasetLoader(data_folder, split).load()


def load_hotpotqa_data(data_folder: str, split: Union[List[str], str] = "train"):
    return HotpotDatasetLoader(data_folder, split).load()


def load_two_wiki_data(data_folder: str, split: Union[List[str], str] = "train"):
    return TwoWikiDatasetLoader(data_folder, split).load()

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

def load_yaml_config(config_path: str, args: argparse.Namespace) -> dict:
    with open(config_path, "r") as fin:
        yaml_config: dict = yaml.safe_load(fin)

    return yaml_config

DATASET_TO_SPLIT_LIST: Dict[str, List[str]] = {
    "nq": ["train", "validation"],
    "triviaqa": ["train", "validation"],
    "hotpotqa": ["train", "dev"],
    "two_wiki": ["train", "dev"],
    "popqa": ["test"],
    "webqa": ["train", "test"],
    "musique": ["train", "dev"],
}


def check_dataset_split(dataset: str, split: str) -> None:
    assert dataset in DATASET_TO_SPLIT_LIST.keys(), f"Dataset {dataset} not found in predefined `DATASET_TO_SPLIT_LIST`"
    assert split in DATASET_TO_SPLIT_LIST[dataset], (
        f"Dataset {dataset} do not have split {split} in `DATASET_TO_SPLIT_LIST`"
    )
    return

def create_dirs(root_dir: str, datasets: List[str]) -> None:
    # Create saving directories if not exist.
    if not os.path.exists(root_dir):
        print(f"Create directory for dataset saving: {root_dir}")
        os.makedirs(root_dir, exist_ok=True)

    for dataset in datasets:
        dataset_dir = os.path.join(root_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

    return


def get_split_filepath(root_dir: str, dataset: str, split: str, sample_num: Optional[int]) -> str:
    if sample_num is None:
        filepath = os.path.join(root_dir, dataset, f"{split}.jsonl")
    else:
        filepath = os.path.join(root_dir, dataset, f"{split}_{sample_num}.jsonl")
    return filepath

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
                        llm_rate = [metric_result == 1 for metric_result in metric_results].count(True) / len(metric_results)
                        organized_results[group_name][metrics_type]["choice"] = llm_rate

    return organized_results

if __name__ == "__main__":
    b = [4, 17, 52, 6]
    a = [3,4,17,73,5,8,10,6]
    print(merge_by_reverse_removal(a, b))