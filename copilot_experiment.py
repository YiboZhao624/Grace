import os
import json
from tqdm import tqdm
from typing import List, Dict, Any
import matplotlib.pyplot as plt

from configs import ChunkerConfig, RetrieverConfig
from chunker import get_chunker
from retriever import get_retriever
from utils import setup_logging, load_raw_qasper_data
from data_generation import QASPERDataGenerator

logger = setup_logging("Retriever_Evaluation")

RETRIEVER_CONFIGS_TO_TEST = [
    RetrieverConfig(retriever_type="bm25", k1=1.5, b=0.75),
    RetrieverConfig(
        retriever_type="vllm",
        model_name="/root/shared_planing/LLM_model/Qwen3-Embedding-0.6B",
        base_url="http://localhost:8001"
    ),
    RetrieverConfig(retriever_type="random"),
]

# --- 在这里配置您的分块器 ---
# 使用与您数据生成时相同的分块器以保证结果可比
CHUNKER_CONFIG = ChunkerConfig(
    chunking_strategy="token",
    model_name="/root/shared_planing/LLM_model/Qwen3-Embedding-0.6B",
    max_length=256,
    overlap=64,
)

DATA_FOLDER = "./data/qasper"
SPLIT_TO_EVALUATE = "test"
RECALL_N_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def plot_results(results: Dict[str, Dict[str, float]], output_filename: str = "copilot/retriever_recall_comparison.png"):
    """
    saving the results of the retriever.

    Args:
        results: a dictionary containing the recall rates of the retriever.
        output_filename: the name of the output image.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']

    for i, (retriever_name, scores) in enumerate(results.items()):
        n_values = sorted([int(k.split('@')[1]) for k in scores.keys()])
        recall_scores = [scores[f'Recall@{n}'] for n in n_values]
        
        ax.plot(n_values, recall_scores, marker=markers[i % len(markers)], linestyle='-', label=retriever_name)

    ax.set_title('Retriever Performance Comparison: Recall@N', fontsize=16)
    ax.set_xlabel('N (Top-K Retrieved Documents)', fontsize=12)
    ax.set_ylabel('Recall Rate', fontsize=12)
    
    ax.set_xticks(RECALL_N_VALUES)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.savefig(output_filename, dpi=300)
    logger.info(f"Performance curve saved to {output_filename}")


def calculate_recall(retrieved_ids: List[int], gt_ids: List[int]) -> bool:
    """
    checking if the retrieved ids contain at least one gt id.
    """
    return len(set(retrieved_ids) & set(gt_ids)) > 0

def evaluate_retriever_performance():
    """
    main evaluation function, for loading data, iterating over the retriever, calculating Recall@N and calling the plotting function.
    """
    chunker = get_chunker(CHUNKER_CONFIG)

    logger.info("Loading and pre-processing QASPER data...")
    from configs import DataGeneratorConfigs
    temp_configs = DataGeneratorConfigs(data_folder=DATA_FOLDER, split=[SPLIT_TO_EVALUATE], method="retriever")
    data_handler = QASPERDataGenerator(chunker, None, temp_configs)
    
    qa_data = data_handler.QA_data[SPLIT_TO_EVALUATE]
    paper_data = data_handler.paper_data[SPLIT_TO_EVALUATE]
    logger.info(f"Data loaded. Found {len(qa_data)} questions and {len(paper_data)} papers for split '{SPLIT_TO_EVALUATE}'.")

    all_results = {}
    for retriever_config in RETRIEVER_CONFIGS_TO_TEST:
        retriever = get_retriever(retriever_config)
        retriever_name = retriever.retriever_type
        logger.info(f"\n===== Evaluating Retriever: {retriever_name} =====")

        hits_at_n = {n: 0 for n in RECALL_N_VALUES}
        total_questions = 0

        for qa in tqdm(qa_data, desc=f"Processing {retriever_name}"):
            paper_id = qa["paper_id"]
            question = qa["question"]
            
            if paper_id not in paper_data:
                logger.warning(f"Paper ID {paper_id} not found in paper_data. Skipping question.")
                continue

            chunks = paper_data[paper_id]["chunks"]
            if not chunks:
                logger.warning(f"No chunks found for paper {paper_id}. Skipping question.")
                continue

            gt_evidence_ids, _ = data_handler._get_gt_evidence(qa, SPLIT_TO_EVALUATE)
            
            if not gt_evidence_ids:
                continue
            
            total_questions += 1
            
            retriever.reset()
            retriever.index(chunks)
            max_n = max(RECALL_N_VALUES)
            _, retriever_res = retriever.retrieve(question, max_n)
            retrieved_ids = [idx for idx, _ in retriever_res]
            
            for n in RECALL_N_VALUES:
                top_n_retrieved = retrieved_ids[:n]
                if calculate_recall(top_n_retrieved, gt_evidence_ids):
                    hits_at_n[n] += 1
        
        recall_at_n_scores = {f"Recall@{n}": hits_at_n[n] / total_questions if total_questions > 0 else 0 
                              for n in RECALL_N_VALUES}
        all_results[retriever_name] = recall_at_n_scores
    

    logger.info("\n\n===== Final Retriever Performance Results =====")
    print(json.dumps(all_results, indent=4))
    
    with open("copilot/retriever_evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    logger.info("Results saved to copilot/retriever_evaluation_results.json")
    
    plot_results(all_results)


if __name__ == "__main__":
    evaluate_retriever_performance()