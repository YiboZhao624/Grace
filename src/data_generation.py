
'''
this file is used to generate the training data for the model.
mainly include the following steps:
1. load the training set of QASPER, including the QA and paper data.
2. choose a chunking strategy to chunk the paper data, which should be implemented in the chunker.py file.
3. choose the distractor strategy to generate the distractor evidence, it should be within the "retriever", and "random".
    ps: retriever means using the identified retriever to retrieve top k+1 evidence from the chunked paper data.
        if the ground truth evidence chunk is not retrieved, the top k retrieved chunks will be added as distractor evidence.
        else, the ground truth will be removed from the top k+1, and the others will serve as distractor evidence.
4. for each QA pair
    4.1 first map the ground truth evidence with the chunk id.
    4.2 add the distractor evidence.
    4.3 fit the data into the template.
    4.4 tag with the data generation type.

the options for this file include:
    1. chunking strategy: imported from the chunker.py file.
    2. distractor strategy: retriever or random
    3. top-k: the number of total evidence to be input.
    4. Fake evidence rate: how many entries include no ground truth evidence.
    5. output path: the path to save the generated data.

Notice: if the distractor strategy is "retriever", the retriever should be imported from the retriever.py file.
'''
# the required data format for the veRL:
# {
#     "data_source": str, align with the custom_reward.py
#     "prompt": [
#         {
#             "role": "user" or "assistant",
#             "content": str
#         }
#     ],
#     "ability": "ICL",
#     "reward_model": {"style": "rule",
#                      "ground_truth": {"answer": str,
#                                       "evidence": List[str]}},
#     "extra_info":{
#         "split": str,
#         "index": int,
#         "question_id": str,
#         "paper_id": str,
#         "generation_type": str,
#         "generation_config": List[int]
#     }
# }

import argparse
import json
import random
import copy
import logging
from numbers import Integral
from typing import List, Dict, Any, Tuple, Optional, Union, Set

import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import (
    load_raw_qasper_data,
    load_hotpotqa_data,
    load_two_wiki_data,
    merge_by_reverse_removal,
)
from prompts import Prompt_templates
from configs import DataGeneratorConfigs, PreprocessConfig, RetrieverConfig, ChunkerConfig, RerankerConfig, DataGeneratorConfig
from copy import deepcopy

from chunker import *
from retriever import *
from reranker import *

# random.seed(42)


def find_evidence_chunks(evidence_texts: List[str], chunks: List[str], threshold: float = 0.8) -> List[int]:
    """
    find the chunk indices that contain the evidence.
    
    Args:
        evidence_texts: the list of ground truth evidence texts.
        chunks: the list of chunks.
        threshold: the threshold for matching.
    
    Returns:
        the list of chunk indices that contain the evidence.

    Notes:
        The recall threshold is used to filter out the larger chunks that contain a small part of the evidence, which does not contribute to the answer.
        The precision is designed for smaller chunks that may cannot recall the whole evidence itself, but it is a part of the evidence.
        This is designed for mitigating the impact of the fragmentary chunked evidence.
    """
    evidence_chunk_ids = set()
    
    for evidence in evidence_texts:
        if not evidence.strip():
            continue
            
        # simple substring matching
        for i, chunk in enumerate(chunks):
            if evidence.strip() in chunk or chunk in evidence.strip():
                evidence_chunk_ids.add(i)
                continue
            
            # calculate the overlap rate of the words.
            evidence_words = set(evidence.lower().split())
            chunk_words = set(chunk.lower().split())
            
            if len(evidence_words) > 0:
                recall = len(evidence_words & chunk_words) / len(evidence_words)
                precision = len(evidence_words & chunk_words) / len(chunk_words)
                if recall >= threshold or precision >= threshold:
                    evidence_chunk_ids.add(i)
    
    return list(evidence_chunk_ids)




def _load_dataset_for_generator(
    dataset: str,
    data_folder: str,
    split: Union[List[str], str],
) -> Tuple[Dict[str, Dict[str, Dict]], Dict[str, List[Dict]]]:
    dataset = dataset.lower()
    if dataset == "qasper":
        return load_raw_qasper_data(data_folder, split)
    if dataset == "hotpotqa":
        return load_hotpotqa_data(data_folder, split)
    if dataset in {"2wiki", "2wikimultihop", "2wikimultihopqa"}:
        return load_two_wiki_data(data_folder, split)
    raise ValueError(f"Unsupported dataset '{dataset}' for data generation")


class BaseDatasetGenerator:
    DATASET_KEY: Optional[str] = None

    def __init__(
        self,
        chunker: Chunker,
        retriever: Optional[Retriever],
        config: DataGeneratorConfig,
        *,
        reranker: Optional[Reranker] = None,
    ) -> None:
        self.config = config
        self.chunker = chunker
        self.retriever = retriever
        self.reranker = reranker
        self.data_folder = config.data_folder
        self.dataset_key = (self.DATASET_KEY or config.dataset).lower()
        self.paper_data: Dict[str, Dict[str, Dict]] = {}
        self.QA_data: Dict[str, List[Dict]] = {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}:{self.dataset_key}")
        self._load_all_splits(config.split)

    @staticmethod
    def _normalize_splits(splits: Union[List[str], str]) -> List[str]:
        if isinstance(splits, str):
            return [splits]
        return list(splits)

    def _load_all_splits(self, splits: Union[List[str], str]) -> None:
        normalized = self._normalize_splits(splits)
        paper_data, qa_data = _load_dataset_for_generator(self.dataset_key, self.data_folder, normalized)
        self.paper_data = paper_data
        self.QA_data = qa_data
        self.ensure_full_text()
        self._ensure_chunks()

    def ensure_full_text(self) -> None:
        for split, papers in self.paper_data.items():
            for paper_id, paper in papers.items():
                full_text = paper.get("full_text")
                if isinstance(full_text, list):
                    flattened: List[str] = []
                    for section in full_text:
                        if isinstance(section, dict):
                            paragraphs = section.get("paragraphs", [])
                            flattened.extend(
                                paragraph.strip()
                                for paragraph in paragraphs
                                if isinstance(paragraph, str) and paragraph.strip()
                            )
                        elif isinstance(section, str) and section.strip():
                            flattened.append(section.strip())
                    paper["full_text"] = "\n\n".join(flattened)
                elif isinstance(full_text, str):
                    paper["full_text"] = full_text.strip()
                else:
                    paper["full_text"] = ""

    def _ensure_chunks(self) -> None:
        for split, papers in self.paper_data.items():
            for paper_id, paper in papers.items():
                per_context_chunks: List[str] = []
                contexts = paper.get("contexts") or []
                if contexts:
                    for context in contexts:
                        title = context.get("title", "") if isinstance(context, dict) else ""
                        text = context.get("text", "") if isinstance(context, dict) else ""
                        combined = "\n".join(part.strip() for part in [title, text] if part and part.strip())
                        combined = combined.strip()
                        if not combined:
                            continue
                        chunks = self.chunker.chunk(combined)
                        per_context_chunks.extend(chunks)
                else:
                    text = paper.get("full_text", "")
                    if not isinstance(text, str):
                        text = ""
                    per_context_chunks = self.chunker.chunk(text)
                paper["chunks"] = per_context_chunks

    def _prepare_entry_metadata(self, qa_data: Dict, split: str) -> Dict:
        entry: Dict[str, Any] = {
            "data_source": self.dataset_key.upper(),
            "ability": "ICL",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "gt_evidence": [""],
                    "answer": [""],
                },
            },
            "extra_info": {
                "paper_id": qa_data.get("paper_id", ""),
                "question_id": qa_data.get("question_id", ""),
                "question": qa_data.get("question", ""),
                "split": split,
                "references": qa_data.get("references", []),
                "data_source": self.dataset_key.upper(),
            },
        }
        return entry

    def _convert_evidence_to_chunk_ids(self, gt_evidence: List[str], paper_id: str, split: str) -> List[int]:
        evidence_chunk_ids: set[int] = set()
        use_overlap_matching = self.dataset_key == "qasper"
        paper_chunks = self.paper_data[split][paper_id]["chunks"]

        for evidence in gt_evidence:
            if not isinstance(evidence, str):
                continue
            evidence_text = evidence.strip()
            if not evidence_text:
                continue

            for idx, chunk in enumerate(paper_chunks):
                if not isinstance(chunk, str):
                    continue
                chunk_text = chunk.strip()
                if not chunk_text:
                    continue

                if evidence_text in chunk_text or chunk_text in evidence_text:
                    evidence_chunk_ids.add(idx)
                    continue

                if not use_overlap_matching:
                    continue

                evidence_words = set(evidence_text.lower().split())
                chunk_words = set(chunk_text.lower().split())
                if not evidence_words or not chunk_words:
                    continue

                overlap = evidence_words & chunk_words
                recall = len(overlap) / len(evidence_words)
                precision = len(overlap) / len(chunk_words)
                if (
                    recall >= self.config.recall_threshold
                    or precision >= self.config.precision_threshold
                ):
                    evidence_chunk_ids.add(idx)

        return list(evidence_chunk_ids)

    def _get_gt_evidence(self, qa_data: Dict, split: str) -> Tuple[List[int], List[str]]:
        gt_evidence_idx: List[int] = []
        gt_evidence_text: List[str] = []
        for reference in qa_data.get("references", []):
            evidence_list = reference.get("evidence", [])
            gt_evidence_text.extend(evidence_list)
            gt_evidence_idx.extend(
                self._convert_evidence_to_chunk_ids(evidence_list, qa_data["paper_id"], split)
            )
        return list(set(gt_evidence_idx)), list(set(gt_evidence_text))

    @staticmethod
    def _get_gt_answer(qa_data: Dict) -> List[str]:
        answers = {reference.get("answer", "") for reference in qa_data.get("references", [])}
        return [answer for answer in answers if answer]

    def manage_chunk_text_list_to_text(self, chunk_text_list: List[str]) -> str:
        if self.config.prompt_template == "TrustAlign_hotpotqa":
            return "\n\n".join(f"Document [{idx + 1}]: {text}" for idx, text in enumerate(chunk_text_list))
        if not self.config.number_template:
            return "\n\n".join(chunk_text_list)
        return "\n\n".join(f"{idx + 1}.{text}" for idx, text in enumerate(chunk_text_list))

    def _requires_full_gt_coverage(self) -> bool:
        return self.dataset_key in {"hotpotqa", "2wikimultihop"}

    def _effective_top_k(self, gt_evidence_ids: List[int]) -> int:
        '''
        For QASPER, it is the original top_k, as any gt evidence is enough to support the answer.
        For the HotpotQA and 2WikiMultihop, it is the max of the original top_k and the number of unique gt evidence ids. As we need all of the gt evidence ids to support the answer.
        '''
        if not gt_evidence_ids:
            return self.config.top_k
        unique_count = len({idx for idx in gt_evidence_ids if isinstance(idx, int)})
        if self._requires_full_gt_coverage():
            return max(self.config.top_k, unique_count)
        if self.dataset_key == "qasper":
            return max(1, self.config.top_k)
        return self.config.top_k

    @staticmethod
    def _truncate_preserving_gt(order: List[int], limit: int, gt_set: Set[int]) -> List[int]:
        if limit <= 0:
            return []
        if len(order) <= limit:
            deduped: List[int] = []
            seen: Set[int] = set()
            for idx in order:
                if idx in seen:
                    continue
                deduped.append(idx)
                seen.add(idx)
            return deduped
        result: List[int] = []
        seen: Set[int] = set()
        required_order: List[int] = []
        required_seen: Set[int] = set()
        for idx in order:
            if idx in gt_set and idx not in required_seen:
                required_order.append(idx)
                required_seen.add(idx)
        for idx in order:
            if idx in seen:
                continue
            if len(result) >= limit:
                break
            remaining_slots = limit - len(result)
            remaining_required = [req for req in required_order if req not in result]
            if idx in gt_set:
                result.append(idx)
                seen.add(idx)
            else:
                if remaining_slots > len(remaining_required):
                    result.append(idx)
                    seen.add(idx)
        for idx in required_order:
            if idx not in result:
                if len(result) < limit:
                    result.append(idx)
                else:
                    for pos in range(len(result) - 1, -1, -1):
                        if result[pos] not in gt_set:
                            result[pos] = idx
                            break
        final: List[int] = []
        final_seen: Set[int] = set()
        for idx in result:
            if idx in final_seen:
                continue
            final.append(idx)
            final_seen.add(idx)
            if len(final) >= limit:
                break
        return final

    def _merge_candidate_with_gt(self, candidate_ids: List[int], gt_evidence_ids: List[int], limit: int) -> List[int]:
        gt_set: Set[int] = {idx for idx in gt_evidence_ids if isinstance(idx, int)}
        seen: Set[int] = set()
        merged: List[int] = []

        def _append_unique(source: List[int]) -> None:
            for idx in source:
                if not (isinstance(idx, int) or isinstance(idx, np.int64)):
                    continue
                if idx in seen:
                    continue
                merged.append(idx)
                seen.add(idx)

        _append_unique(candidate_ids)
        _append_unique(gt_evidence_ids)

        effective_limit = limit
        if effective_limit is None or effective_limit <= 0:
            effective_limit = max(len(gt_set), len(merged))

        if effective_limit > 0 and len(merged) > effective_limit:
            merged = self._truncate_preserving_gt(merged, effective_limit, gt_set)

        if effective_limit > 0 and len(merged) > effective_limit:
            merged = merged[:effective_limit]

        if not merged and candidate_ids:
            merged = candidate_ids[:effective_limit] if effective_limit > 0 else candidate_ids

        return merged

    def _maybe_prepare_train_flags(self, split: str) -> None:
        if split != "train":
            return
        qa_list = self.QA_data.get(split, [])
        if not qa_list:
            return
        answerable: List[Dict] = []
        unanswerable: List[Dict] = []
        for qa_data in qa_list:
            if any(ref.get("answer") == "Unanswerable" for ref in qa_data.get("references", [])):
                unanswerable.append(qa_data)
            else:
                answerable.append(qa_data)
        target_without_gt = int(len(qa_list) * self.config.wo_gt_evidence_rate) - len(unanswerable)
        if target_without_gt < 0:
            target_without_gt = 0
        if target_without_gt > len(answerable):
            target_without_gt = len(answerable)
        sampled = random.sample(answerable, target_without_gt) if target_without_gt > 0 else []
        sampled_ids = {qa.get("question_id") for qa in sampled}
        for qa_data in qa_list:
            is_unanswerable = any(ref.get("answer") == "Unanswerable" for ref in qa_data.get("references", []))
            qa_data["gt_evidence"] = not (is_unanswerable or qa_data.get("question_id") in sampled_ids)
        total_wo_gt = len(unanswerable) + len(sampled)
        self.logger.info(
            "Total entries w/o ground truth evidence: %s (Naturally Unanswerable: %s, Sampled: %s)",
            total_wo_gt,
            len(unanswerable),
            len(sampled),
        )

    def generate_for_split(self, split: str, answer_balance: bool = True) -> List[Dict]:
        qa_list = self.QA_data.get(split, [])
        if not qa_list:
            self.logger.warning("No QA data found for split '%s' in dataset '%s'", split, self.dataset_key)
            return []
        self._maybe_prepare_train_flags(split)
        examples: List[Dict] = []
        prev_paper_id: Optional[str] = None
        for qa_data in tqdm(qa_list, desc=f"Generating data ({self.dataset_key}-{split})"):
            if self.retriever is not None and qa_data.get("paper_id") != prev_paper_id:
                self.retriever.reset()
                self.retriever.index(self.paper_data[split][qa_data["paper_id"]]["chunks"])
                prev_paper_id = qa_data.get("paper_id")
            entry = self._prepare_entry_metadata(qa_data, split)
            gt_evidence_ids, gt_evidence_text = self._get_gt_evidence(qa_data, split)
            entry["reward_model"]["ground_truth"]["answer"] = self._get_gt_answer(qa_data)
            evidence_chunks, entry = self.organize_evidence(qa_data, gt_evidence_ids, gt_evidence_text, entry, split, answer_balance)
            evidence_input = self.manage_chunk_text_list_to_text(evidence_chunks)
            prompt_template = copy.deepcopy(Prompt_templates[self.config.prompt_template])
            prompt_template[-1]["content"] = (
                prompt_template[-1]["content"]
                .replace("{question}", qa_data.get("question", ""))
                .replace("{ref}", evidence_input)
            )
            entry["prompt"] = prompt_template
            examples.append(entry)
        return examples

    def generate_data_for_splits(self) -> Dict[str, List[Dict]]:
        results: Dict[str, List[Dict]] = {}
        for split in self._normalize_splits(self.config.split):
            results[f"{split}_{self.config.method}"] = self.generate_for_split(split, answer_balance=self.config.answer_balance)
        return results

    def organize_evidence(
        self,
        qa_data: Dict,
        gt_evidence_ids: List[int],
        gt_evidence_text: List[str],
        entry: Dict,
        split: str,
    ) -> Tuple[List[str], Dict]:
        raise NotImplementedError("Subclasses must implement organize_evidence")


class QasperDatasetMixin:
    DATASET_KEY = "qasper"

    def ensure_full_text(self) -> None:  # type: ignore[override]
        for split, papers in self.paper_data.items():
            for paper_id, paper in papers.items():
                text_parts: List[str] = []
                title = paper.get("title")
                abstract = paper.get("abstract")
                if isinstance(title, str) and title.strip():
                    text_parts.append(title.strip())
                if isinstance(abstract, str) and abstract.strip():
                    text_parts.append(abstract.strip())
                contexts = paper.get("full_text") or []
                for context in contexts:
                    ctx_title = context.get("section_name", "") if isinstance(context, dict) else ""
                    ctx_text = context.get("paragraphs", []) if isinstance(context, dict) else []
                    to_combine = [ctx_title] + ctx_text
                    combined = "\n".join(part.strip() for part in to_combine if part and part.strip())
                    if combined:
                        text_parts.append(combined)
                paper["full_text"] = "\n\n".join(text_parts)
        super().ensure_full_text()


class HotpotDatasetMixin:
    DATASET_KEY = "hotpotqa"

    def ensure_full_text(self) -> None:  # type: ignore[override]
        super().ensure_full_text()


class TwoWikiDatasetMixin:
    DATASET_KEY = "2wikimultihop"

    def ensure_full_text(self) -> None:  # type: ignore[override]
        super().ensure_full_text()


class RetrieverDataGenerator(BaseDatasetGenerator):
    def __init__(self, chunker: Chunker, retriever: Retriever, config: DataGeneratorConfig) -> None:
        if retriever is None:
            raise ValueError("RetrieverDataGenerator requires a retriever instance")
        super().__init__(chunker, retriever, config)

    def organize_evidence(
        self,
        qa_data: Dict,
        gt_evidence_ids: List[int],
        gt_evidence_text: List[str],
        entry: Dict,
        split: str,
        answer_balance: bool = True
    ) -> Tuple[List[str], Dict]:
        assert self.retriever is not None
        top_k_for_gt = self._effective_top_k(gt_evidence_ids)
        retrieve_k = max(self.config.top_k + 3, top_k_for_gt)
        evidence_texts, retriever_res = self.retriever.retrieve(qa_data["question"], retrieve_k)
        retrieved_ids = [idx for idx, _ in retriever_res][: len(self.paper_data[split][qa_data["paper_id"]]["chunks"])]
        if split == "train" and answer_balance:
            entry["reward_model"]["ground_truth"]["gt_evidence_retrieved"] = None
            if not qa_data.get("gt_evidence", True):
                entry["extra_info"]["generation_type"] = "wo_gt_evidence"
                final_evidence_ids = [idx for idx in retrieved_ids if idx not in gt_evidence_ids][: self.config.top_k]
                entry["extra_info"]["gt_evidence_chunk_ids"] = []
                entry["extra_info"]["evidence_chunk_ids"] = final_evidence_ids
                entry["extra_info"]["distractor_chunk_ids"] = final_evidence_ids
                entry["reward_model"]["ground_truth"]["gt_evidence"] = [""]
            else:
                entry["extra_info"]["generation_type"] = "gt_evidence"
                merged_ids = merge_by_reverse_removal(retrieved_ids, gt_evidence_ids)
                final_evidence_ids = self._merge_candidate_with_gt(merged_ids, gt_evidence_ids, top_k_for_gt)
                entry["extra_info"]["gt_evidence_chunk_ids"] = gt_evidence_ids
                entry["extra_info"]["evidence_chunk_ids"] = final_evidence_ids
                entry["extra_info"]["distractor_chunk_ids"] = [idx for idx in merged_ids if idx not in final_evidence_ids]
                entry["reward_model"]["ground_truth"]["gt_evidence"] = gt_evidence_text
            evidence_chunks = [
                self.paper_data[split][qa_data["paper_id"]]["chunks"][chunk_idx]
                for chunk_idx in entry["extra_info"]["evidence_chunk_ids"]
            ]
        elif split == "train" and not answer_balance:
            entry["reward_model"]["ground_truth"]["gt_evidence_retrieved"] = None
            top_ids = retrieved_ids[:self.config.top_k]
            entry["extra_info"]["evidence_chunk_ids"] = top_ids
            entry["extra_info"]["distractor_chunk_ids"] = [idx for idx in top_ids if idx not in gt_evidence_ids]
            entry["extra_info"]["gt_evidence_chunk_ids"] = gt_evidence_ids
            entry["reward_model"]["ground_truth"]["gt_evidence"] = gt_evidence_text
            if self._requires_full_gt_coverage() and gt_evidence_ids:
                is_retrieved_gt_evidence = all(
                    evidence_id in top_ids for evidence_id in gt_evidence_ids
                )
            else:
                is_retrieved_gt_evidence = any(
                    evidence_id in top_ids for evidence_id in gt_evidence_ids
                )
            if is_retrieved_gt_evidence:
                entry["reward_model"]["ground_truth"]["gt_evidence"] = gt_evidence_text
            else:
                entry["reward_model"]["ground_truth"]["gt_evidence"] = [""]
            evidence_chunks = [
                self.paper_data[split][qa_data["paper_id"]]["chunks"][chunk_idx]
                for chunk_idx in top_ids
            ]
        else:
            entry["extra_info"]["generation_type"] = "retrieved"
            top_ids = retrieved_ids[:self.config.top_k]
            entry["extra_info"]["evidence_chunk_ids"] = top_ids
            entry["extra_info"]["distractor_chunk_ids"] = [idx for idx in top_ids if idx not in gt_evidence_ids]
            entry["extra_info"]["gt_evidence_chunk_ids"] = gt_evidence_ids
            entry["reward_model"]["ground_truth"]["gt_evidence"] = gt_evidence_text
            if self._requires_full_gt_coverage() and gt_evidence_ids:
                entry["reward_model"]["ground_truth"]["gt_evidence_retrieved"] = all(
                    evidence_id in top_ids for evidence_id in gt_evidence_ids
                )
            else:
                entry["reward_model"]["ground_truth"]["gt_evidence_retrieved"] = any(
                    evidence_id in top_ids for evidence_id in gt_evidence_ids
                )
            evidence_chunks = [
                self.paper_data[split][qa_data["paper_id"]]["chunks"][chunk_idx]
                for chunk_idx in top_ids
            ]
        return evidence_chunks, entry


class OracleDataGenerator(BaseDatasetGenerator):
    def __init__(self, chunker: Chunker, config: DataGeneratorConfig) -> None:
        super().__init__(chunker, None, config)

    def organize_evidence(
        self,
        qa_data: Dict,
        gt_evidence_ids: List[int],
        gt_evidence_text: List[str],
        entry: Dict,
        split: str,
    ) -> Tuple[List[str], Dict]:
        entry["extra_info"]["generation_type"] = "gt_evidence"
        entry["extra_info"]["gt_evidence_chunk_ids"] = gt_evidence_ids
        entry["extra_info"]["distractor_chunk_ids"] = []
        entry["reward_model"]["ground_truth"]["gt_evidence"] = gt_evidence_text
        if split != "train":
            entry["reward_model"]["ground_truth"]["gt_evidence_retrieved"] = True
        chunks = self.paper_data[split][qa_data["paper_id"]]["chunks"]
        seen: Set[int] = set()
        evidence_chunks: List[str] = []
        for chunk_idx in gt_evidence_ids:
            if chunk_idx in seen or not (0 <= chunk_idx < len(chunks)):
                continue
            seen.add(chunk_idx)
            evidence_chunks.append(chunks[chunk_idx])
        if not evidence_chunks:
            evidence_chunks = gt_evidence_text
        return evidence_chunks, entry


class RetrieverRerankerDataGenerator(BaseDatasetGenerator):
    def __init__(
        self,
        chunker: Chunker,
        retriever: Retriever,
        reranker: Reranker,
        config: DataGeneratorConfig,
    ) -> None:
        if retriever is None:
            raise ValueError("RetrieverRerankerDataGenerator requires a retriever instance")
        if reranker is None:
            raise ValueError("RetrieverRerankerDataGenerator requires a reranker instance")
        super().__init__(chunker, retriever, config, reranker=reranker)

    def organize_evidence(
        self,
        qa_data: Dict,
        gt_evidence_ids: List[int],
        gt_evidence_text: List[str],
        entry: Dict,
        split: str,
        answer_balance: bool = True
    ) -> Tuple[List[str], Dict]:
        assert self.retriever is not None and self.reranker is not None
        top_k_for_gt = self._effective_top_k(gt_evidence_ids)
        retrieve_k = max(self.config.top_k + 5, top_k_for_gt)
        evidence_texts, retriever_res = self.retriever.retrieve(qa_data["question"], retrieve_k)
        retrieved_ids = [idx for idx, _ in retriever_res]
        if split == "train" and answer_balance:
            candidate_ids = merge_by_reverse_removal(retrieved_ids, gt_evidence_ids)
            candidate_ids = self._merge_candidate_with_gt(
                candidate_ids,
                gt_evidence_ids,
                max(retrieve_k, top_k_for_gt),
            )
        elif split == "train" and not answer_balance:
            candidate_ids = retrieved_ids[:self.config.top_k]
        else:
            seen_ids: Set[int] = set()
            candidate_ids = []
            for idx in retrieved_ids:
                if not isinstance(idx, Integral):
                    continue
                if idx in seen_ids:
                    continue
                candidate_ids.append(idx)
                seen_ids.add(idx)
                if len(candidate_ids) >= retrieve_k:
                    break
        document_chunks = self.paper_data[split][qa_data["paper_id"]]["chunks"]
        valid_candidate_ids: List[int] = []
        candidate_chunks: List[str] = []
        for chunk_idx in candidate_ids:
            if not isinstance(chunk_idx, Integral):
                continue
            idx_int = int(chunk_idx)
            if not (0 <= idx_int < len(document_chunks)):
                continue
            valid_candidate_ids.append(idx_int)
            candidate_chunks.append(document_chunks[idx_int])
        reranked_index, reranked_evidence = self.reranker.rerank(candidate_chunks, qa_data["question"])
        reranked_ids = [valid_candidate_ids[idx] for idx in reranked_index]
        self.logger.debug("Reranked chunk ids: %s", reranked_ids)
        if split == "train" and answer_balance:
            entry["reward_model"]["ground_truth"]["gt_evidence_retrieved"] = None
            if not qa_data.get("gt_evidence", True):
                entry["extra_info"]["generation_type"] = "wo_gt_evidence"
                final_ids = [idx for idx in reranked_ids if idx not in gt_evidence_ids][: self.config.top_k]
                entry["extra_info"]["gt_evidence_chunk_ids"] = []
                entry["extra_info"]["distractor_chunk_ids"] = final_ids
                entry["extra_info"]["evidence_chunk_ids"] = final_ids
                entry["reward_model"]["ground_truth"]["gt_evidence"] = [""]
            else:
                entry["extra_info"]["generation_type"] = "gt_evidence"
                final_ids = self._merge_candidate_with_gt(reranked_ids, gt_evidence_ids, top_k_for_gt)
                entry["extra_info"]["evidence_chunk_ids"] = final_ids
                entry["extra_info"]["distractor_chunk_ids"] = [idx for idx in reranked_ids if idx not in final_ids]
                entry["extra_info"]["gt_evidence_chunk_ids"] = gt_evidence_ids
                entry["reward_model"]["ground_truth"]["gt_evidence"] = gt_evidence_text
            evidence_chunks = [
                self.paper_data[split][qa_data["paper_id"]]["chunks"][chunk_idx]
                for chunk_idx in entry["extra_info"]["evidence_chunk_ids"]
            ]
        elif split == "train" and not answer_balance:
            entry["reward_model"]["ground_truth"]["gt_evidence_retrieved"] = None
            seen: Set[int] = set()
            top_ids: List[int] = []
            for idx in reranked_ids:
                if idx in seen:
                    continue
                top_ids.append(idx)
                seen.add(idx)
                if top_k_for_gt > 0 and len(top_ids) >= top_k_for_gt:
                    break
            entry["extra_info"]["evidence_chunk_ids"] = top_ids
            entry["extra_info"]["distractor_chunk_ids"] = [idx for idx in top_ids if idx not in gt_evidence_ids]
            entry["extra_info"]["gt_evidence_chunk_ids"] = gt_evidence_ids
            # entry["reward_model"]["ground_truth"]["gt_evidence"] = gt_evidence_text
            if self._requires_full_gt_coverage() and gt_evidence_ids:
                is_retrieved_gt_evidence = all(
                    evidence_id in top_ids for evidence_id in gt_evidence_ids
                )
            else:
                is_retrieved_gt_evidence = any(
                    evidence_id in top_ids for evidence_id in gt_evidence_ids
                )
            if is_retrieved_gt_evidence:
                entry["reward_model"]["ground_truth"]["gt_evidence"] = gt_evidence_text
            else:
                entry["reward_model"]["ground_truth"]["gt_evidence"] = [""]
            evidence_chunks = [
                self.paper_data[split][qa_data["paper_id"]]["chunks"][chunk_idx]
                for chunk_idx in top_ids
            ]
        else:
            entry["extra_info"]["generation_type"] = "retrieved"
            seen: Set[int] = set()
            top_ids: List[int] = []
            for idx in reranked_ids:
                if idx in seen:
                    continue
                top_ids.append(idx)
                seen.add(idx)
                if top_k_for_gt > 0 and len(top_ids) >= top_k_for_gt:
                    break
            entry["extra_info"]["evidence_chunk_ids"] = top_ids
            entry["extra_info"]["distractor_chunk_ids"] = [idx for idx in top_ids if idx not in gt_evidence_ids]
            entry["extra_info"]["gt_evidence_chunk_ids"] = gt_evidence_ids
            entry["reward_model"]["ground_truth"]["gt_evidence"] = gt_evidence_text
            if self._requires_full_gt_coverage() and gt_evidence_ids:
                self.logger.debug("%s eval requires full GT coverage", self.dataset_key)
                entry["reward_model"]["ground_truth"]["gt_evidence_retrieved"] = all(
                    evidence_id in top_ids for evidence_id in gt_evidence_ids
                )
            else:
                self.logger.debug("%s eval allows partial GT coverage", self.dataset_key)
                entry["reward_model"]["ground_truth"]["gt_evidence_retrieved"] = any(
                    evidence_id in top_ids for evidence_id in gt_evidence_ids
                )
            evidence_chunks = [
                self.paper_data[split][qa_data["paper_id"]]["chunks"][chunk_idx]
                for chunk_idx in top_ids
            ]
        return evidence_chunks, entry


class QasperRetrieverDataGenerator(QasperDatasetMixin, RetrieverDataGenerator):
    pass


class QasperOracleDataGenerator(QasperDatasetMixin, OracleDataGenerator):
    pass


class QasperRetrieverRerankerDataGenerator(QasperDatasetMixin, RetrieverRerankerDataGenerator):
    pass


class HotpotRetrieverDataGenerator(HotpotDatasetMixin, RetrieverDataGenerator):
    pass


class HotpotOracleDataGenerator(HotpotDatasetMixin, OracleDataGenerator):
    pass


class HotpotRetrieverRerankerDataGenerator(HotpotDatasetMixin, RetrieverRerankerDataGenerator):
    pass


class TwoWikiRetrieverDataGenerator(TwoWikiDatasetMixin, RetrieverDataGenerator):
    pass


class TwoWikiOracleDataGenerator(TwoWikiDatasetMixin, OracleDataGenerator):
    pass


class TwoWikiRetrieverRerankerDataGenerator(TwoWikiDatasetMixin, RetrieverRerankerDataGenerator):
    pass


def _canonical_dataset_key(dataset: str) -> str:
    key = dataset.lower()
    if key in {"2wikimultihop", "2wikimultihopqa", "2wiki"}:
        return "2wikimultihop"
    return key


def get_data_generator(config: DataGeneratorConfig):
    method = config.method
    dataset_key = _canonical_dataset_key(config.dataset)
    chunker = get_chunker(config.chunker_config)

    registry: Dict[Tuple[str, str], Any] = {
        ("qasper", "retriever"): QasperRetrieverDataGenerator,
        ("qasper", "random"): QasperRetrieverDataGenerator,
        ("qasper", "oracle"): QasperOracleDataGenerator,
        ("qasper", "retriever_reranker"): QasperRetrieverRerankerDataGenerator,
        ("hotpotqa", "retriever"): HotpotRetrieverDataGenerator,
        ("hotpotqa", "random"): HotpotRetrieverDataGenerator,
        ("hotpotqa", "oracle"): HotpotOracleDataGenerator,
        ("hotpotqa", "retriever_reranker"): HotpotRetrieverRerankerDataGenerator,
        ("2wikimultihop", "retriever"): TwoWikiRetrieverDataGenerator,
        ("2wikimultihop", "random"): TwoWikiRetrieverDataGenerator,
        ("2wikimultihop", "oracle"): TwoWikiOracleDataGenerator,
        ("2wikimultihop", "retriever_reranker"): TwoWikiRetrieverRerankerDataGenerator,
    }

    key = (dataset_key, method)
    if key not in registry:
        supported = ", ".join(sorted({f"{ds}:{m}" for ds, m in registry}))
        raise ValueError(f"Unsupported dataset/method combination '{config.dataset}/{method}'. Supported: {supported}")

    generator_cls = registry[key]

    if issubclass(generator_cls, RetrieverRerankerDataGenerator):
        retriever = get_retriever(config.retriever_config)
        reranker = get_reranker(config.reranker_config)
        return generator_cls(chunker, retriever, reranker, config)

    if issubclass(generator_cls, RetrieverDataGenerator):
        retriever = get_retriever(config.retriever_config)
        return generator_cls(chunker, retriever, config)

    if issubclass(generator_cls, OracleDataGenerator):
        return generator_cls(chunker, config)

    raise ValueError(f"Generator class '{generator_cls.__name__}' is not supported in factory")