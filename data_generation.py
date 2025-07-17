
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

import argparse
import json
import random
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from chunker import *
from retriever import *
import numpy as np
from tqdm import tqdm
from utils import load_raw_qasper_data
from prompts import Prompt_templates
from configs import DataGeneratorConfig

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
    # the implemented generators include: 
    # 1. QASPERRetrieverDataGenerator: using the retriever.
    # 2. QASPERRandomDataGenerator: using the random strategy.
    # 3. QASPERRetrieverRerankerGenerator: using the retriever and reranker.
    # 4. QASPERRandomRerankerGenerator: using the retriever and reranker and random strategy.
    # 5. QASPEROracleDataGenerator: using the ground truth evidence.
    def __init__(self, chunker: Chunker, retriever: Union[Retriever, None], paper_path: str, qa_path: str, config: Dict = None):
        self.config = config
        self.chunker = chunker
        self.retriever = retriever
        self.paper_path = paper_path
        self.qa_path = qa_path
        self.paper_data = None
        self.QA_data = None
        self.load_data()

    def load_data(self) -> Tuple[Dict, Dict]:
        paper_data, QA_data = load_raw_qasper_data(self.paper_path, self.qa_path)
        self.paper_data = paper_data
        self.extract_full_text()
        for paper_id, paper_data in self.paper_data.items():
            chunks = self.chunk_paper(paper_id)
            self.paper_data[paper_id]["chunks"] = chunks
        self.QA_data = QA_data

    def generate(self) -> List[Dict]:
        raise NotImplementedError("You can't directly use the meta class. Subclasses must implement this method")

    def chunk_paper(self, paper_id: str) -> List[str]:
        """
        chunk the paper data.
        """
        full_text = self.paper_data[paper_id]["full_text"]
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
        for paper_id, paper_data in self.paper_data.items():    
            text_parts = []        
            if paper_data.get("title"):
                text_parts.append(paper_data["title"])
            if paper_data.get("abstract"):
                text_parts.append(paper_data["abstract"])
            
            if "full_text" in paper_data:
                for section in paper_data["full_text"]:
                    if "paragraphs" in section:
                        for paragraph in section["paragraphs"]:
                            if paragraph.strip():
                                text_parts.append(paragraph.strip())
            full_text = "\n\n".join(text_parts)
            self.paper_data[paper_id]["full_text"] = full_text
    
    def _convert_evidence_to_chunk_ids(self, gt_evidence: List[str], paper_id: str) -> List[int]:
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
            for i, chunk in enumerate(self.paper_data[paper_id]["chunks"]):
                if evidence.strip() in chunk or chunk in evidence.strip():
                    evidence_chunk_ids.add(i)
                    continue
                
                # calculate the overlap rate of the words.
                evidence_words = set(evidence.lower().split())
                chunk_words = set(chunk.lower().split())
                
                if len(evidence_words) > 0:
                    recall = len(evidence_words & chunk_words) / len(evidence_words)
                    precision = len(evidence_words & chunk_words) / len(chunk_words)
                    if recall >= self.config.get("recall_threshold", 0.5) or precision >= self.config.get("precision_threshold", 0.5):
                        evidence_chunk_ids.add(i)
        
        return list(evidence_chunk_ids)

    def _prepare_entry_metadata(self, qa_data: Dict) -> Dict:
        '''
        prepare the metadata for the entry.
        '''
        entry = {}
        entry["data_source"] = "QASPER"
        entry["ability"] = "ICL"
        entry["reward_model"] = {"style": "rule", "ground_truth": {}}
        entry["extra_info"] = {}
        entry["extra_info"]["paper_id"] = qa_data["paper_id"]
        entry["extra_info"]["question_id"] = qa_data.get("question_id", "")
        entry["extra_info"]["question"] = qa_data.get("question", "")
        entry["extra_info"]["split"] = self.qa_path
        entry["extra_info"]["references"] = qa_data.get("references", [])
        return entry
    
    def _get_gt_evidence_ids(self, qa_data: Dict) -> List[int]:
        '''
        process the ground truth evidence, because there are several annotators for each QA pair, we take the union of the evidence chunk ids.
        '''
        gt_evidence_idx = []
        for annotator_idx in range(len(qa_data["references"])):
            gt_evidence_ids = self._convert_evidence_to_chunk_ids(qa_data["references"][annotator_idx]["evidence"], qa_data["paper_id"])
            gt_evidence_idx.extend(gt_evidence_ids)
        gt_evidence_idx = list(set(gt_evidence_idx))
        return gt_evidence_idx


class QASPERRetrieverDataGenerator(QASPERDataGenerator):
    def __init__(self, chunker: Chunker, retriever: Retriever, paper_path: str, qa_path: str, config: DataGeneratorConfig):
        super().__init__(chunker, retriever, paper_path, qa_path, config)
        self.retriever = retriever
        self.chunker = chunker
        self.paper_path = paper_path
        self.qa_path = qa_path
        self.config = config
        self.QA_data: List[Dict] = []
        self.paper_data: Dict[str, Dict] = {}
        self.load_data()

    def generate(self) -> List[Dict]:
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

        examples = []
        prev_paper_id = None
        for qa_data in self.QA_data:
            # initialize the retriever.
            if prev_paper_id != qa_data["paper_id"]:
                self.retriever.reset()
                self.retriever.index(self.paper_data[qa_data["paper_id"]]["chunks"])
                prev_paper_id = qa_data["paper_id"]

            
            # copy the basic and extra information.
            entry = self._prepare_entry_metadata(qa_data)
            
            # get the ground truth evidence chunk ids.
            gt_evidence_idx = self._get_gt_evidence_ids(qa_data)

            # no matter what, retrieve the top k+n chunks first, here, we assume that the number of ground truth evidence chunks won't be larger than 3.
            evidence_texts, retriever_res = self.retriever.retrieve(qa_data["question"], top_k + 3)
            retrieved_ids = [idx for idx, _ in retriever_res]

            # decide whether to create a fake evidence sample.
            if random.random() < wo_gt_evidence_rate:
                entry["extra_info"]["generation_type"] = "wo_gt_evidence"
                # choose the evidence not in ground truth.
                final_evidence_ids = [idx for idx in retrieved_ids if idx not in gt_evidence_idx]
                final_evidence_ids = final_evidence_ids[:top_k]
                # record the evidence chunk ids.
                entry["extra_info"]["gt_evidence_chunk_ids"] = []
                entry["extra_info"]["distractor_chunk_ids"] = final_evidence_ids
            else:
                entry["extra_info"]["generation_type"] = "gt_evidence"
                final_evidence_ids = gt_evidence_idx + [idx for idx in retrieved_ids if idx not in gt_evidence_idx]
                final_evidence_ids = final_evidence_ids[:top_k]
                entry["extra_info"]["evidence_chunk_ids"] = final_evidence_ids
                entry["extra_info"]["distractor_chunk_ids"] = [idx for idx in retrieved_ids if idx not in final_evidence_ids]

            # change the evidence chunk ids to chunk content.
            # then fit the content into the template.
            evidence_chunks = []
            for chunk_idx in final_evidence_ids:
                evidence_chunks.append(self.paper_data[qa_data["paper_id"]]["chunks"][chunk_idx])
            
            if self.config.number_template:
                evidence_input = "\n\n".join(evidence_chunks)
            else:
                evidence_input = "\n\n".join(f"{i+1}.{s}" for i, s in enumerate(evidence_chunks))
            
            # fit the evidence into the template.
            prompt_template = Prompt_templates[self.config.prompt_template]
            prompt_template = prompt_template.format(evidence=evidence_input, question=qa_data["question"], answer=qa_data["references"][0]["answer"])
            entry["prompt"] = prompt_template
            examples.append(entry)
        return examples