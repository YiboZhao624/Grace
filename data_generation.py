
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
import pandas as pd
import copy
from typing import List, Dict, Any, Tuple, Optional
from chunker import *
from retriever import *
from reranker import *
import numpy as np
from tqdm import tqdm
from utils import load_raw_qasper_data, merge_by_reverse_removal, safe_len
from prompts import Prompt_templates
from configs import DataGeneratorConfigs, PreprocessConfig, RetrieverConfig, ChunkerConfig, RerankerConfig, DataGeneratorConfig
from copy import deepcopy

random.seed(42)


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




class QASPERDataGenerator:
    # this class is used to generate the training data for the QASPER dataset.
    # it is a meta class that should be inherited by the specific data generator.
    # to implement a new data generator, just inherit this class and implement the organize_evidence function.
    # the implemented generators include: 
    # 1. QASPERRetrieverDataGenerator: using the retriever.
    # 2. QASPERRandomDataGenerator: using the random strategy.
    # 3. QASPERRetrieverRerankerGenerator: using the retriever and reranker.
    # 4. QASPERRandomRerankerGenerator: using the retriever and reranker and random strategy.
    # 5. QASPEROracleDataGenerator: using the ground truth evidence.
    def __init__(self, chunker: Chunker, retriever: Union[Retriever, None], config: DataGeneratorConfig):
        # self.configs (复数) 不再需要，现在只有一个 self.config (单数)
        self.config = config
        self.chunker = chunker
        self.retriever = retriever
        self.data_folder = config.data_folder # 从单数 config 获取
        self.paper_data: Dict[str, Dict[str, Dict]] = {}
        self.QA_data:Dict[str, List[Dict]] = {}
        # load_data 方法现在需要知道要加载哪些 split
        self.load_data(config.split) 

    def load_data(self, splits: List[str]) -> None:
        paper_data, QA_data = load_raw_qasper_data(self.data_folder, splits)
        self.paper_data = paper_data
        self.extract_full_text()
        for split, paper_data in self.paper_data.items():
            for paper_id, one_paper_data in paper_data.items():
                chunks = self.chunk_paper(split, paper_id)
                self.paper_data[split][paper_id]["chunks"] = chunks
        self.QA_data = QA_data

    def chunk_paper(self, split:str, paper_id: str) -> List[str]:
        """
        chunk the paper data.
        """
        full_text = self.paper_data[split][paper_id]["full_text"]
        chunks = self.chunker.chunk(full_text)
        return chunks

    def extract_full_text(self):
        """
        extract the full text from the paper data.
        Args:
            paper_data: the data of a single paper.
        Returns:
            context after concatenating the title, abstract and full text.
        """
        for split, paper_data in self.paper_data.items():
            for paper_id, one_paper_data in paper_data.items():    
                text_parts = []        
                if one_paper_data.get("title"):
                    text_parts.append(one_paper_data["title"])
                if one_paper_data.get("abstract"):
                    text_parts.append(one_paper_data["abstract"])
            
                if "full_text" in one_paper_data:
                    for section in one_paper_data["full_text"]:
                        if "paragraphs" in section:
                            for paragraph in section["paragraphs"]:
                                if paragraph.strip():
                                    text_parts.append(paragraph.strip())
                full_text = "\n\n".join(text_parts)
                self.paper_data[split][paper_id]["full_text"] = full_text
    
    def _convert_evidence_to_chunk_ids(self, gt_evidence: List[str], paper_id: str, split: str) -> List[int]:
        """
        find the chunk indices that contain the evidence.
        
        Args:
            gt_evidence: the list of ground truth evidence texts.
            paper_id: the id of the paper.
        
        Returns:
            the list of chunk indices that contain the evidence.

        Notes:
            The recall threshold is used to filter out the larger chunks that contain a small part of the evidence, which does not contribute to the answer.
            The precision is designed for smaller chunks that may cannot recall the whole evidence itself, but it is a part of the evidence.
            This is designed for mitigating the impact of the fragmentary chunked evidence.
            The threshold is set to 0.5, which means that if the recall or precision is greater than 0.5, the chunk is considered as containing the evidence.
        """
        evidence_chunk_ids = set()
        
        for evidence in gt_evidence:
            if not evidence.strip():
                continue
                
            # simple substring matching
            for i, chunk in enumerate(self.paper_data[split][paper_id]["chunks"]):
                if evidence.strip() in chunk or chunk in evidence.strip():
                    evidence_chunk_ids.add(i)
                    continue
                
                # calculate the overlap rate of the words.
                evidence_words = set(evidence.lower().split())
                chunk_words = set(chunk.lower().split())
                
                if len(evidence_words) > 0:
                    recall = len(evidence_words & chunk_words) / len(evidence_words)
                    precision = len(evidence_words & chunk_words) / len(chunk_words)
                    if recall >= self.config.recall_threshold or precision >= self.config.precision_threshold:
                        evidence_chunk_ids.add(i)
        
        return list(evidence_chunk_ids)

    def _prepare_entry_metadata(self, qa_data: Dict) -> Dict:
        '''
        prepare the metadata for the entry.
        '''
        entry = {}
        entry["data_source"] = "QASPER"
        entry["ability"] = "ICL"
        entry["reward_model"] = {"style": "rule", "ground_truth": {"gt_evidence": [""], "answer": [""]}}
        entry["extra_info"] = {}
        entry["extra_info"]["paper_id"] = qa_data["paper_id"]
        entry["extra_info"]["question_id"] = qa_data.get("question_id", "")
        entry["extra_info"]["question"] = qa_data.get("question", "")
        entry["extra_info"]["split"] = self.config.split
        entry["extra_info"]["references"] = qa_data.get("references", [])
        return entry
    
    def _get_gt_evidence(self, qa_data: Dict, split: str) -> Tuple[List[int], List[str]]:
        '''
        process the ground truth evidence, because there are several annotators for each QA pair, we take the union of the evidence chunk ids.
        Although we evaluate the model chose evidence on the token level, we still return the chunk ids.
        This is designed to check if the retrieved chunks includes the ground truth evidence.
        '''
        gt_evidence_idx = []
        gt_evidence_text = []
        for annotator_idx in range(len(qa_data["references"])):
            gt_evidence_text.extend(qa_data["references"][annotator_idx]["evidence"])
            gt_evidence_ids = self._convert_evidence_to_chunk_ids(qa_data["references"][annotator_idx]["evidence"], qa_data["paper_id"], split)
            gt_evidence_idx.extend(gt_evidence_ids)
        gt_evidence_idx = list(set(gt_evidence_idx))
        gt_evidence_text = list(set(gt_evidence_text))
        return gt_evidence_idx, gt_evidence_text

    def _get_gt_answer(self, qa_data: Dict) -> List[str]:
        gt_answer = []
        for annotator_idx in range(len(qa_data["references"])):
            gt_answer.append(qa_data["references"][annotator_idx]["answer"])
        gt_answer = list(set(gt_answer))
        return gt_answer

    def organize_evidence(self, qa_data: Dict, gt_evidence_ids: List[int], gt_evidence_text: List[str], entry: dict) -> Tuple[List[str], dict]:
        '''
        Note: the retriever has already been indexed (if the retriever is required.)
        You need to provide the following information:
        1. get the evidences, and return the evidences in a list of str.
        2. provide the basic info in the entry.
        '''
        raise NotImplementedError("You can't directly use the meta class. Subclasses must implement this method")

    def manage_chunk_text_list_to_text(self, chunk_text_list: List[str]) -> str:
        if self.config.number_template:
            evidence_input = "\n\n".join(chunk_text_list)
        else:
            evidence_input = "\n\n".join(f"{i+1}.{s}" for i, s in enumerate(chunk_text_list))
        return evidence_input

    def generate_single(self) -> List[Dict]:
        # 0. about the data: paper is well chunked, retriever is initialized.
        # 1. manage the configs, important configs include:
        # 1.1 top-k: the number of total evidence to be input, default is 5.
        # 1.2 wo_gt_evidence_rate: the rate of entries without ground truth evidence.
        # 2. for each QA pair:
        # 2.1 decide whether to create a fake evidence sample.
        # 2.2 if it is a fake evidence sample, only retrieve the top k + 1 chunks, if it contains the ground truth evidence, remove it and use the rest as distractor evidence. else, use the top k chunks as distractor evidence.
        # 2.3 if it is a normal evidence sample, retrieve the top k chunks, if it contains the ground truth evidence, directly use it. else, use the top k-1 as distractor evidence.
        # 2.4 fit the data into the template.
        # 2.5 tag with the data generation type.
        # 3. save the data.
        top_k = self.config.top_k
        wo_gt_evidence_rate = self.config.wo_gt_evidence_rate
        if self.config.split == "train":
            answerable_qa = []
            unanswerable_qa = []
            for qa_data in self.QA_data[self.config.split]:
                # check if the answer is "Unanswerable"
                is_unanswerable = any(ref.get("answer") == "Unanswerable" for ref in qa_data.get("references", []))
                if is_unanswerable:
                    unanswerable_qa.append(qa_data)
                else:
                    answerable_qa.append(qa_data)

            # 2. randomly sample "wo_gt_evidence" from answerable_qa
            num_to_sample = int(len(self.QA_data[self.config.split]) * wo_gt_evidence_rate) - len(unanswerable_qa)
            sampled_wo_gt_entries = random.sample(answerable_qa, num_to_sample)
            
            # create a set of question_id for fast lookup
            sampled_wo_gt_set = {qa['question_id'] for qa in sampled_wo_gt_entries}
            
            # 3. iterate over all data, finally determine the gt_evidence flag for each sample
            for qa in self.QA_data[self.config.split]:
                # use set for fast lookup
                is_sampled_as_wo_gt = qa.get('question_id') in sampled_wo_gt_set
                is_naturally_unanswerable = any(ref.get("answer") == "Unanswerable" for ref in qa.get("references", []))

                if is_naturally_unanswerable or is_sampled_as_wo_gt:
                    qa["gt_evidence"] = False
                else:
                    qa["gt_evidence"] = True
            
            total_wo_gt_count = len(unanswerable_qa) + num_to_sample
            logger.info(
                f"Total entries w/o ground truth evidence: {total_wo_gt_count} "
                f"(Naturally Unanswerable: {len(unanswerable_qa)}, "
                f"Sampled from Answerable: {num_to_sample})"
            )

        examples = []
        prev_paper_id = None
        for qa_data in tqdm(self.QA_data[self.config.split], desc="Generating data"):
            # initialize the retriever, if needed.
            if self.retriever is not None and prev_paper_id != qa_data["paper_id"]:
                self.retriever.reset()
                self.retriever.index(self.paper_data[self.config.split][qa_data["paper_id"]]["chunks"])
                prev_paper_id = qa_data["paper_id"]
            
            # copy the basic and extra information.
            entry = self._prepare_entry_metadata(qa_data)
            
            # get the ground truth evidence chunk ids.
            gt_evidence_ids, gt_evidence_text = self._get_gt_evidence(qa_data, self.config.split)
            gt_answer_text = self._get_gt_answer(qa_data)
            entry["reward_model"]["ground_truth"]["answer"] = gt_answer_text

            # organize the evidence.
            evidence_chunks, entry = self.organize_evidence(qa_data, gt_evidence_ids, gt_evidence_text, entry)
            
            evidence_input = self.manage_chunk_text_list_to_text(evidence_chunks)

            # fit the evidence into the template.
            prompt_template = copy.deepcopy(Prompt_templates[self.config.prompt_template])
            # Use replace method to avoid KeyError from curly braces in evidence text
            content = prompt_template[1]["content"]
            content = content.replace("{question}", qa_data["question"])
            content = content.replace("{ref}", evidence_input)
            prompt_template[1]["content"] = content
            entry["prompt"] = prompt_template
            examples.append(entry)
        return examples

    def generate_data_for_splits(self) -> Dict[str, List[Dict]]:
        """
        为配置中定义的所有 split（如 train, val, test）生成数据。
        """
        all_examples = {}
        
        # self.config.split is a list, e.g., ["train", "val", "test"]
        for split in self.config.split:
            original_split = self.config.split 
            self.config.split = split
            
            logger.info(f"Generating for split='{split}', method='{self.config.method}'")
            examples = self.generate_single()

            all_examples[f"{split}_{self.config.method}"] = examples
            
            self.config.split = original_split
            
        return all_examples


class QASPERRetrieverDataGenerator(QASPERDataGenerator):
    def __init__(self, chunker: Chunker, retriever: Retriever, config: DataGeneratorConfigs):
        super().__init__(chunker, retriever, config)
        # Note: self.QA_data and self.paper_data are already initialized by parent class

    def organize_evidence(self, qa_data: Dict, gt_evidence_ids: List[int], gt_evidence_text: List[str], entry: dict) -> Tuple[List[str], dict]:
        # decide whether to create a fake evidence sample.
        # no matter what, retrieve the top k+n chunks first, here, we assume that the number of ground truth evidence chunks won't be larger than 3.
        evidence_texts, retriever_res = self.retriever.retrieve(qa_data["question"], self.config.top_k + 3)
        retrieved_ids = [idx for idx, _ in retriever_res][:len(self.paper_data[self.config.split][qa_data["paper_id"]]["chunks"])]
        if self.config.split == "train":
            if qa_data["gt_evidence"] == False:
                entry["extra_info"]["generation_type"] = "wo_gt_evidence"
                # choose the evidence not in ground truth.
                final_evidence_ids = [idx for idx in retrieved_ids if idx not in gt_evidence_ids]
                final_evidence_ids = final_evidence_ids[:self.config.top_k]
                # record the evidence chunk ids.
                entry["extra_info"]["gt_evidence_chunk_ids"] = []
                entry["extra_info"]["evidence_chunk_ids"] = final_evidence_ids
                entry["extra_info"]["distractor_chunk_ids"] = final_evidence_ids
                entry["reward_model"]["ground_truth"]["gt_evidence"] = [""]
                
            else:
                entry["extra_info"]["generation_type"] = "gt_evidence"
                final_evidence_ids = merge_by_reverse_removal(retrieved_ids, gt_evidence_ids)
                final_evidence_ids = final_evidence_ids[:self.config.top_k]
                entry["extra_info"]["gt_evidence_chunk_ids"] = gt_evidence_ids
                entry["extra_info"]["evidence_chunk_ids"] = final_evidence_ids
                entry["extra_info"]["distractor_chunk_ids"] = [idx for idx in retrieved_ids if idx not in final_evidence_ids]
                entry["reward_model"]["ground_truth"]["gt_evidence"] = gt_evidence_text

            # change the evidence chunk ids to chunk content.
            # then fit the content into the template.
            evidence_chunks = []
            for chunk_idx in final_evidence_ids:
                evidence_chunks.append(self.paper_data[self.config.split][qa_data["paper_id"]]["chunks"][chunk_idx])
        else: # the evidence is for the dev and test.
            entry["extra_info"]["generation_type"] = "retrieved"
            entry["extra_info"]["evidence_chunk_ids"] = retrieved_ids[:self.config.top_k]
            entry["extra_info"]["distractor_chunk_ids"] = []
            entry["reward_model"]["ground_truth"]["gt_evidence"] = gt_evidence_text
            # change the evidence chunk ids to chunk content.
            evidence_chunks = [self.paper_data[self.config.split][qa_data["paper_id"]]["chunks"][chunk_idx] for chunk_idx in retrieved_ids[:self.config.top_k]]
        return evidence_chunks, entry


class QASPEROracleDataGenerator(QASPERDataGenerator):
    '''
    this class is used to generate the data with the oracle evidence.
    It is treated as the most easily type of data.
    '''
    def __init__(self, chunker: Chunker, retriever: Retriever, config: DataGeneratorConfigs):
        super().__init__(chunker, retriever, config)
        # Note: self.QA_data and self.paper_data are already initialized by parent class

    def organize_evidence(self, qa_data: Dict, gt_evidence_ids: List[int], gt_evidence_text: List[str], entry: dict) -> Tuple[List[str], dict]:
        # no need to retrieve the evidence, just use the ground truth evidence.
        entry["extra_info"]["generation_type"] = "gt_evidence"
        entry["extra_info"]["gt_evidence_chunk_ids"] = gt_evidence_ids
        entry["extra_info"]["distractor_chunk_ids"] = []
        entry["reward_model"]["ground_truth"]["gt_evidence"] = gt_evidence_text
        # due to there is no distractor evidence, we directly return the evidence text.
        return gt_evidence_text, entry


class QASPERRetrieverRerankerDataGenerator(QASPERDataGenerator):
    '''
    this class is used to generate the data with the retriever and reranker.
    '''
    def __init__(self, chunker: Chunker, retriever: Retriever, reranker: Reranker, config: DataGeneratorConfigs):
        super().__init__(chunker, retriever, config)
        self.reranker = reranker
        # Note: self.QA_data and self.paper_data are already initialized by parent class

    def organize_evidence(self, qa_data: Dict, gt_evidence_ids: List[int], gt_evidence_text: List[str], entry: dict) -> Tuple[List[str], dict]:
        # retrieve and rerank the evidence.
        evidence_texts, retriever_res = self.retriever.retrieve(qa_data["question"], self.config.top_k + 3)
        retrieved_ids = [idx for idx, _ in retriever_res]
        evidence_chunks = [self.paper_data[self.config.split][qa_data["paper_id"]]["chunks"][chunk_idx] for chunk_idx in retrieved_ids]
        reranked_index, reranked_evidence = self.reranker.rerank(evidence_chunks, qa_data["question"])
        reranked_ids = [retrieved_ids[idx] for idx in reranked_index]
        logger.debug(f"the reranked ids are: {reranked_ids}")
        logger.debug(f"the reranked evidence are: {reranked_evidence}")

        if self.config.split == "train":
            # decide if the ground truth evidence is included in the retrieved evidence.
            if qa_data["gt_evidence"] == False:
                entry["extra_info"]["generation_type"] = "wo_gt_evidence"
                # choose the evidence not in ground truth.
                final_evidence_ids = [idx for idx in reranked_ids if idx not in gt_evidence_ids]
                final_evidence_ids = final_evidence_ids[:self.config.top_k]
                # record the evidence chunk ids.
                entry["extra_info"]["gt_evidence_chunk_ids"] = []
                entry["extra_info"]["distractor_chunk_ids"] = final_evidence_ids
                entry["extra_info"]["evidence_chunk_ids"] = final_evidence_ids
                entry["reward_model"]["ground_truth"]["gt_evidence"] = [""]
                
            else:
                entry["extra_info"]["generation_type"] = "gt_evidence"
                # merge the gt_evidence_index and the reranked_ids, make sure the order is correct.
                final_evidence_ids = merge_by_reverse_removal(reranked_ids, gt_evidence_ids)
                final_evidence_ids = final_evidence_ids[:self.config.top_k]
                entry["extra_info"]["evidence_chunk_ids"] = final_evidence_ids
                entry["extra_info"]["distractor_chunk_ids"] = [idx for idx in reranked_ids if idx not in final_evidence_ids]
                entry["reward_model"]["ground_truth"]["gt_evidence"] = gt_evidence_text

            # change the evidence chunk ids to chunk content.
            # then fit the content into the template.
            evidence_chunks = [self.paper_data[self.config.split][qa_data["paper_id"]]["chunks"][chunk_idx] for chunk_idx in final_evidence_ids]
        else: # the evidence is for the dev and test.
            entry["extra_info"]["generation_type"] = "retrieved"
            entry["extra_info"]["evidence_chunk_ids"] = retrieved_ids[:self.config.top_k]
            entry["extra_info"]["distractor_chunk_ids"] = []
            entry["reward_model"]["ground_truth"]["gt_evidence"] = gt_evidence_text
            # change the evidence chunk ids to chunk content.
            evidence_chunks = [self.paper_data[self.config.split][qa_data["paper_id"]]["chunks"][chunk_idx] for chunk_idx in retrieved_ids[:self.config.top_k]]
        return evidence_chunks, entry


def get_data_generator(config: DataGeneratorConfig): # 参数变为单数的 config
    """
    根据单一、明确的配置获取数据生成器实例。
    """
    method = config.method
    chunker = get_chunker(config.chunker_config)

    if method == "retriever":
        retriever = get_retriever(config.retriever_config)
        return QASPERRetrieverDataGenerator(chunker, retriever, config)
    elif method == "retriever_reranker":
        retriever = get_retriever(config.retriever_config)
        reranker = get_reranker(config.reranker_config)
        return QASPERRetrieverRerankerDataGenerator(chunker, retriever, reranker, config)
    elif method == "oracle":
        # Oracle 不需要 retriever，所以传入 None
        return QASPEROracleDataGenerator(chunker, None, config)
    elif method == "random":
        # Random retriever 也通过 get_retriever 获取
        retriever = get_retriever(config.retriever_config)
        return QASPERRetrieverDataGenerator(chunker, retriever, config)
    else:
        raise ValueError(f"Unsupported data generator method: {method}")