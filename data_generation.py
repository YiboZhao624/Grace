
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
from chunker import get_chunker
from retriever import get_retriever

def extract_full_text(paper_data: Dict) -> str:
    """
    从paper数据中提取完整文本
    
    Args:
        paper_data: 单个paper的数据
    
    Returns:
        拼接后的完整文本
    """
    text_parts = []
    
    # 添加标题和摘要
    if paper_data.get("title"):
        text_parts.append(paper_data["title"])
    if paper_data.get("abstract"):
        text_parts.append(paper_data["abstract"])
    
    # 添加正文
    if "full_text" in paper_data:
        for section in paper_data["full_text"]:
            if "paragraphs" in section:
                for paragraph in section["paragraphs"]:
                    if paragraph.strip():
                        text_parts.append(paragraph.strip())
    
    return "\n\n".join(text_parts)


def find_evidence_chunks(evidence_texts: List[str], chunks: List[str], threshold: float = 0.8) -> List[int]:
    """
    找到包含evidence的chunk索引
    
    Args:
        evidence_texts: ground truth evidence文本列表
        chunks: 文档chunks列表
        threshold: 匹配阈值
    
    Returns:
        包含evidence的chunk索引列表
    """
    evidence_chunk_ids = set()
    
    for evidence in evidence_texts:
        if not evidence.strip():
            continue
            
        # 简单的子串匹配
        for i, chunk in enumerate(chunks):
            if evidence.strip() in chunk or chunk in evidence.strip():
                evidence_chunk_ids.add(i)
                continue
            
            # 计算词汇重叠率
            evidence_words = set(evidence.lower().split())
            chunk_words = set(chunk.lower().split())
            
            if len(evidence_words) > 0:
                overlap = len(evidence_words & chunk_words) / len(evidence_words)
                if overlap >= threshold:
                    evidence_chunk_ids.add(i)
    
    return list(evidence_chunk_ids)


def generate_random_distractors(chunks: List[str], evidence_chunk_ids: List[int], num_distractors: int) -> List[int]:
    """
    随机生成distractor chunk索引
    
    Args:
        chunks: 所有chunks
        evidence_chunk_ids: ground truth evidence chunk索引
        num_distractors: 需要的distractor数量
    
    Returns:
        distractor chunk索引列表
    """
    available_ids = [i for i in range(len(chunks)) if i not in evidence_chunk_ids]
    return random.sample(available_ids, min(num_distractors, len(available_ids)))


def generate_retriever_distractors(
    retriever: Retriever, 
    question: str, 
    chunks: List[str], 
    evidence_chunk_ids: List[int], 
    top_k: int
) -> List[int]:
    """
    使用retriever生成distractor chunk索引
    
    Args:
        retriever: Retriever实例
        question: 问题文本
        chunks: 所有chunks
        evidence_chunk_ids: ground truth evidence chunk索引
        top_k: 总共需要的evidence数量
    
    Returns:
        distractor chunk索引列表
    """
    # 检索top k+1个chunks
    retrieved_chunks = retriever.retrieve(question, top_k + 1)
    
    # 找到检索结果对应的chunk索引
    retrieved_indices = []
    for retrieved_chunk in retrieved_chunks:
        for i, chunk in enumerate(chunks):
            if chunk == retrieved_chunk:
                retrieved_indices.append(i)
                break
    
    # 移除ground truth evidence chunks
    distractor_indices = [idx for idx in retrieved_indices if idx not in evidence_chunk_ids]
    
    # 如果检索结果不包含ground truth，直接返回top k个
    if not any(idx in evidence_chunk_ids for idx in retrieved_indices):
        return distractor_indices[:top_k]
    
    # 否则返回除了ground truth之外的其他chunks
    return distractor_indices


def create_training_example(
    question: str,
    answer: str,
    evidence_chunks: List[str],
    distractor_chunks: List[str],
    generation_type: str,
    paper_id: str,
    question_id: str
) -> Dict:
    """
    创建训练样本
    
    Args:
        question: 问题文本
        answer: 答案文本
        evidence_chunks: evidence chunks列表
        distractor_chunks: distractor chunks列表
        generation_type: 数据生成类型
        paper_id: paper ID
        question_id: 问题ID
    
    Returns:
        格式化的训练样本
    """
    # 简单的模板格式
    all_chunks = evidence_chunks + distractor_chunks
    random.shuffle(all_chunks)  # 随机打乱顺序
    
    context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(all_chunks)])
    
    example = {
        "paper_id": paper_id,
        "question_id": question_id,
        "question": question,
        "answer": answer,
        "context": context,
        "evidence_chunks": evidence_chunks,
        "distractor_chunks": distractor_chunks,
        "generation_type": generation_type,
        "num_evidence": len(evidence_chunks),
        "num_distractors": len(distractor_chunks)
    }
    
    return example


def should_create_fake_evidence(fake_evidence_rate: float) -> bool:
    """
    根据fake evidence rate决定是否创建不包含ground truth的样本
    
    Args:
        fake_evidence_rate: fake evidence比例
    
    Returns:
        是否创建fake evidence样本
    """
    return random.random() < fake_evidence_rate


def process_paper(
    paper_id: str,
    paper_data: Dict,
    chunker: Chunker,
    retriever: Optional[Retriever],
    distractor_strategy: str,
    top_k: int,
    fake_evidence_rate: float
) -> List[Dict]:
    """
    处理单个paper的所有QA pairs
    
    Args:
        paper_id: paper ID
        paper_data: paper数据
        chunker: chunker实例
        retriever: retriever实例
        distractor_strategy: distractor策略
        top_k: 总evidence数量
        fake_evidence_rate: fake evidence比例
    
    Returns:
        训练样本列表
    """
    examples = []
    
    # 提取并chunk文档
    full_text = extract_full_text(paper_data)
    chunks = chunker.chunk(full_text)
    
    if len(chunks) == 0:
        return examples
    
    # 如果使用retriever，建立索引
    if retriever is not None:
        retriever.index(chunks)
    
    # 处理每个QA pair
    for qa in paper_data.get("qas", []):
        question = qa.get("question", "")
        question_id = qa.get("question_id", "")
        
        if not question.strip():
            continue
        
        # 处理每个答案
        for answer_data in qa.get("answers", []):
            answer_obj = answer_data.get("answer", {})
            
            # 跳过无法回答的问题
            if answer_obj.get("unanswerable", False):
                continue
            
            # 获取答案文本
            answer = answer_obj.get("free_form_answer", "")
            if not answer.strip():
                continue
            
            # 获取evidence文本
            evidence_texts = answer_obj.get("evidence", [])
            
            # 找到包含evidence的chunks
            evidence_chunk_ids = find_evidence_chunks(evidence_texts, chunks)
            
            # 决定是否创建fake evidence样本
            create_fake = should_create_fake_evidence(fake_evidence_rate)
            
            if create_fake:
                # 创建不包含ground truth evidence的样本
                if distractor_strategy == "random":
                    distractor_chunk_ids = generate_random_distractors(chunks, [], top_k)
                else:
                    distractor_chunk_ids = generate_retriever_distractors(
                        retriever, question, chunks, [], top_k
                    )
                
                evidence_chunks = []
                distractor_chunks = [chunks[i] for i in distractor_chunk_ids[:top_k]]
                generation_type = f"fake_{distractor_strategy}"
                
            else:
                # 创建包含ground truth evidence的样本
                evidence_chunks = [chunks[i] for i in evidence_chunk_ids]
                num_distractors = max(0, top_k - len(evidence_chunks))
                
                if num_distractors > 0:
                    if distractor_strategy == "random":
                        distractor_chunk_ids = generate_random_distractors(
                            chunks, evidence_chunk_ids, num_distractors
                        )
                    else:
                        distractor_chunk_ids = generate_retriever_distractors(
                            retriever, question, chunks, evidence_chunk_ids, top_k
                        )
                        distractor_chunk_ids = distractor_chunk_ids[:num_distractors]
                    
                    distractor_chunks = [chunks[i] for i in distractor_chunk_ids]
                else:
                    distractor_chunks = []
                
                generation_type = f"normal_{distractor_strategy}"
            
            # 创建训练样本
            example = create_training_example(
                question=question,
                answer=answer,
                evidence_chunks=evidence_chunks,
                distractor_chunks=distractor_chunks,
                generation_type=generation_type,
                paper_id=paper_id,
                question_id=question_id
            )
            
            examples.append(example)
    
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate training data for RLRAG model")
    
    # 数据路径参数
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to QASPER dataset file")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save generated training data")
    
    # Chunking策略参数
    parser.add_argument("--chunking_strategy", type=str, 
                       choices=["sentence", "paragraph", "greedy_sentence", "greedy_paragraph"],
                       default="sentence",
                       help="Chunking strategy to use")
    parser.add_argument("--max_chunk_length", type=int, default=512,
                       help="Maximum chunk length for greedy chunkers")
    
    # Distractor策略参数
    parser.add_argument("--distractor_strategy", type=str,
                       choices=["random", "retriever"],
                       default="random",
                       help="Strategy to generate distractor evidence")
    parser.add_argument("--retriever_type", type=str,
                       choices=["bm25", "sentence_transformer"],
                       default="bm25",
                       help="Type of retriever to use when distractor_strategy is 'retriever'")
    
    # 数据生成参数
    parser.add_argument("--top_k", type=int, default=5,
                       help="Total number of evidence chunks to include")
    parser.add_argument("--fake_evidence_rate", type=float, default=0.1,
                       help="Rate of entries with no ground truth evidence")
    
    # 其他参数
    parser.add_argument("--max_papers", type=int, default=None,
                       help="Maximum number of papers to process (for testing)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--output_format", type=str, choices=["json", "parquet"],
                       default="json",
                       help="Output file format")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    print(f"Loading QASPER data from {args.data_path}...")
    paper_data, QA_data = load_raw_qasper_data(args.data_path)
    
    print(f"Initializing chunker: {args.chunking_strategy}")
    chunker = get_chunker(args.chunking_strategy, args.max_chunk_length)
    
    print(f"Initializing retriever for distractor strategy: {args.distractor_strategy}")
    retriever_config = {"retriever_type": args.retriever_type}
    retriever = get_retriever(args.distractor_strategy, retriever_config)
    
    print("Processing papers...")
    all_examples = []
    
    papers_to_process = list(paper_data.keys())
    if args.max_papers:
        papers_to_process = papers_to_process[:args.max_papers]
    
    for paper_id in tqdm(papers_to_process, desc="Processing papers"):
        examples = process_paper(
            paper_id=paper_id,
            paper_data=paper_data[paper_id],
            chunker=chunker,
            retriever=retriever,
            distractor_strategy=args.distractor_strategy,
            top_k=args.top_k,
            fake_evidence_rate=args.fake_evidence_rate
        )
        all_examples.extend(examples)
    
    print(f"Generated {len(all_examples)} training examples")
    
    # 保存结果
    print(f"Saving results to {args.output_path}...")
    if args.output_format == "json":
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(all_examples, f, ensure_ascii=False, indent=2)
    elif args.output_format == "parquet":
        df = pd.DataFrame(all_examples)
        df.to_parquet(args.output_path, index=False)
    
    # 打印统计信息
    print("\n=== Generation Statistics ===")
    print(f"Total examples: {len(all_examples)}")
    
    generation_types = {}
    for example in all_examples:
        gen_type = example["generation_type"]
        generation_types[gen_type] = generation_types.get(gen_type, 0) + 1
    
    for gen_type, count in generation_types.items():
        print(f"{gen_type}: {count} ({count/len(all_examples)*100:.1f}%)")
    
    avg_evidence = np.mean([ex["num_evidence"] for ex in all_examples])
    avg_distractors = np.mean([ex["num_distractors"] for ex in all_examples])
    print(f"Average evidence chunks per example: {avg_evidence:.2f}")
    print(f"Average distractor chunks per example: {avg_distractors:.2f}")
    
    print("Data generation completed!")


if __name__ == "__main__":
    main()

