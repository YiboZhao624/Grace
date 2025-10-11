import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

from utils import setup_logging


logger = setup_logging("Preprocess")


# QASPER (existing implementation retained)
def process_qasper_data(data: Dict) -> Tuple[List[Dict], Dict[str, Dict]]:
    processed_qa_data: List[Dict] = []
    processed_paper_data: Dict[str, Dict] = {}

    for paper_id, value in data.items():
        processed_paper_data.setdefault(
            paper_id,
            {
                "title": value["title"],
                "abstract": value["abstract"],
                "full_text": value["full_text"],
            },
        )

        for qa_info in value["qas"]:
            references = []
            for annotation_info in qa_info["answers"]:
                answer_info = annotation_info["answer"]

                if answer_info["unanswerable"]:
                    references.append({"answer": "Unanswerable", "evidence": [], "type": "none"})
                    continue

                if answer_info["extractive_spans"]:
                    answer = ", ".join(answer_info["extractive_spans"])
                    answer_type = "extractive"
                elif answer_info["free_form_answer"]:
                    answer = answer_info["free_form_answer"]
                    answer_type = "abstractive"
                elif answer_info["yes_no"]:
                    answer = "Yes"
                    answer_type = "boolean"
                elif answer_info["yes_no"] is not None:
                    answer = "No"
                    answer_type = "boolean"
                else:
                    raise RuntimeError(f"Annotation {answer_info['annotation_id']} does not contain an answer")

                evidence = answer_info["evidence"]
                references.append({"answer": answer, "evidence": evidence, "type": answer_type})

            processed_qa_data.append(
                {
                    "question_id": qa_info["question_id"],
                    "paper_id": paper_id,
                    "question": qa_info["question"],
                    "references": references,
                    "data_source": "QASPER",
                }
            )

    return processed_qa_data, processed_paper_data


def preprocess_qasper(data_folder: str) -> None:
    split_to_file = {
        "train": "qasper-train-v0.3.json",
        "val": "qasper-dev-v0.3.json",
        "test": "qasper-test-v0.3.json",
    }

    outputs: Dict[str, Tuple[List[Dict], Dict]] = {}
    for split, filename in split_to_file.items():
        path = os.path.join(data_folder, filename)
        if not os.path.exists(path):
            logger.warning(f"QASPER {split} file not found: {path}. Skipping.")
            continue

        with open(path, "r", encoding="utf-8") as fin:
            raw_data = json.load(fin)

        outputs[split] = process_qasper_data(raw_data)
        logger.info(
            f"QASPER {split}: QA entries={len(outputs[split][0])}, papers={len(outputs[split][1])}"
        )

    for split, (qa_data, paper_data) in outputs.items():
        split_dir = os.path.join(data_folder, split)
        os.makedirs(split_dir, exist_ok=True)

        with open(os.path.join(split_dir, "QA_data.json"), "w", encoding="utf-8") as fout:
            json.dump(qa_data, fout, indent=2, ensure_ascii=False)
        with open(os.path.join(split_dir, "paper_data.json"), "w", encoding="utf-8") as fout:
            json.dump(paper_data, fout, indent=2, ensure_ascii=False)

    logger.info("Finished preprocessing QASPER.")


# HotpotQA
def _compose_context(paragraphs: List[str]) -> str:
    return " ".join(sentence.strip() for sentence in paragraphs if sentence.strip())


def _get_hotpot_evidence(supporting_facts: List[List], contexts: List[List]) -> List[str]:
    context_map: Dict[str, List[str]] = {title: sentences for title, sentences in contexts}
    evidence_sentences: List[str] = []
    for title, sent_idx in supporting_facts:
        sentences = context_map.get(title, [])
        if 0 <= sent_idx < len(sentences):
            evidence_sentences.append(sentences[sent_idx].strip())
    return evidence_sentences


def _sample_list(data: List[Dict], sample_size: Optional[int], seed: int) -> List[Dict]:
    if sample_size is None:
        return data
    if sample_size <= 0 or sample_size >= len(data):
        return data

    rng = random.Random(seed)
    sampled_indices = rng.sample(range(len(data)), sample_size)
    sampled_indices.sort()
    return [data[idx] for idx in sampled_indices]


def process_hotpotqa_split(data: List[Dict], split: str) -> List[Dict]:
    processed: List[Dict] = []
    for item in data:
        answer = item.get("answer", "")
        answer_type = "boolean" if answer.lower() in {"yes", "no"} else "extractive"
        evidence = _get_hotpot_evidence(item.get("supporting_facts", []), item.get("context", []))

        processed.append(
            {
                "question_id": item.get("_id"),
                "question": item.get("question", ""),
                "data_source": "HotpotQA",
                "references": [
                    {
                        "answer": answer,
                        "evidence": evidence,
                        "type": answer_type,
                    }
                ],
                "metadata": {
                    "level": item.get("level"),
                    "type": item.get("type"),
                    "split": split,
                    "context": [
                        {
                            "title": title,
                            "text": _compose_context(sentences),
                        }
                        for title, sentences in item.get("context", [])
                    ],
                },
            }
        )

    return processed


def preprocess_hotpotqa(
    data_folder: str,
    train_sample_size: Optional[int] = None,
    sample_seed: int = 42,
) -> None:
    split_to_file = {
        "train": "hotpot_train_v1.1.json",
        "dev": "hotpot_dev_distractor_v1.json",
    }

    for split, filename in split_to_file.items():
        path = os.path.join(data_folder, filename)
        if not os.path.exists(path):
            logger.warning(f"HotpotQA {split} file not found: {path}. Skipping.")
            continue

        with open(path, "r", encoding="utf-8") as fin:
            raw_data = json.load(fin)

        if split == "train":
            original_size = len(raw_data)
            raw_data = _sample_list(raw_data, train_sample_size, sample_seed)
            if len(raw_data) != original_size:
                logger.info(
                    f"HotpotQA {split}: sampled {len(raw_data)} entries from {original_size} (seed={sample_seed})"
                )

        processed = process_hotpotqa_split(raw_data, split)
        logger.info(f"HotpotQA {split}: {len(processed)} entries")

        split_dir = os.path.join(data_folder, split)
        os.makedirs(split_dir, exist_ok=True)
        with open(os.path.join(split_dir, "QA_data.json"), "w", encoding="utf-8") as fout:
            json.dump(processed, fout, indent=2, ensure_ascii=False)

    logger.info("Finished preprocessing HotpotQA.")


# 2WikiMultiHopQA
def _format_two_wiki_evidence(evidences: List[List[str]]) -> List[str]:
    formatted: List[str] = []
    for triple in evidences:
        formatted.append(" ".join(part.strip() for part in triple if part))
    return formatted

def process_two_wiki_split(data: List[Dict], split: str) -> List[Dict]:
    processed: List[Dict] = []
    for item in data:
        answer = item.get("answer", "")
        answer_type = "boolean" if answer.lower() in {"yes", "no"} else "extractive"

        processed.append(
            {
                "question_id": item.get("_id"),
                "question": item.get("question", ""),
                "data_source": "2WikiMultiHopQA",
                "references": [
                    {
                        "answer": answer,
                        "evidence": _format_two_wiki_evidence(item.get("evidences", [])),
                        "type": answer_type,
                    }
                ],
                "metadata": {
                    "type": item.get("type"),
                    "split": split,
                    "supporting_facts": item.get("supporting_facts", []),
                    "context": [
                        {
                            "title": title,
                            "text": _compose_context(sentences),
                        }
                        for title, sentences in item.get("context", [])
                    ],
                },
            }
        )

    return processed


def preprocess_two_wiki(
    data_folder: str,
    train_sample_size: Optional[int] = None,
    sample_seed: int = 42,
) -> None:
    split_to_file = {
        "train": "data/train.json",
        "dev": "data/dev.json",
        "test": "data/test.json",
    }

    for split, relative_path in split_to_file.items():
        path = os.path.join(data_folder, relative_path)
        if not os.path.exists(path):
            logger.warning(f"2WikiMultiHopQA {split} file not found: {path}. Skipping.")
            continue

        with open(path, "r", encoding="utf-8") as fin:
            raw_data = json.load(fin)

        if split == "train":
            original_size = len(raw_data)
            raw_data = _sample_list(raw_data, train_sample_size, sample_seed)
            if len(raw_data) != original_size:
                logger.info(
                    f"2WikiMultiHopQA {split}: sampled {len(raw_data)} entries from {original_size} (seed={sample_seed})"
                )

        processed = process_two_wiki_split(raw_data, split)
        logger.info(f"2WikiMultiHopQA {split}: {len(processed)} entries")

        split_dir = os.path.join(data_folder, split)
        os.makedirs(split_dir, exist_ok=True)
        with open(os.path.join(split_dir, "QA_data.json"), "w", encoding="utf-8") as fout:
            json.dump(processed, fout, indent=2, ensure_ascii=False)

    logger.info("Finished preprocessing 2WikiMultiHopQA.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess datasets into unified format")
    parser.add_argument(
        "--dataset",
        choices=["qasper", "hotpotqa", "2wiki", "all"],
        default="all",
        help="Which dataset to preprocess.",
    )
    parser.add_argument(
        "--data-root",
        default="./data",
        help="Root folder where raw datasets are stored.",
    )
    parser.add_argument(
        "--hotpotqa-train-sample-size",
        type=int,
        default=20000,
        help="Maximum HotpotQA train examples to keep. Use non-positive value to keep all.",
    )
    parser.add_argument(
        "--two-wiki-train-sample-size",
        type=int,
        default=10000,
        help="Maximum 2WikiMultiHopQA train examples to keep. Use non-positive value to keep all.",
    )
    parser.add_argument(
        "--train-sample-seed",
        type=int,
        default=42,
        help="Random seed for train sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset in {"qasper", "all"}:
        preprocess_qasper(os.path.join(args.data_root, "qasper"))

    if args.dataset in {"hotpotqa", "all"}:
        preprocess_hotpotqa(
            os.path.join(args.data_root, "hotpot"),
            train_sample_size=args.hotpotqa_train_sample_size,
            sample_seed=args.train_sample_seed,
        )

    if args.dataset in {"2wiki", "all"}:
        preprocess_two_wiki(
            os.path.join(args.data_root, "2wikimultihop"),
            train_sample_size=args.two_wiki_train_sample_size,
            sample_seed=args.train_sample_seed,
        )


if __name__ == "__main__":
    main()