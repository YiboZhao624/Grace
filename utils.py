import json
import os

def load_raw_qasper_data(data_folder: str, split = "train"):
    path = os.path.join(data_folder, split)
    paper_path = os.path.join(path, "paper_data.json")
    QA_path = os.path.join(path, "QA_data.json")
    with open(paper_path, "r") as f:
        paper_data = json.load(f)
    with open(QA_path, "r") as f:
        QA_data = json.load(f)
    return paper_data, QA_data

def get_chunker(chunking_strategy: str, max_length: int = None, overlap: int = None, model_name: str = None) -> Chunker:
    """
    根据策略名称获取chunker实例
    
    Args:
        chunking_strategy: chunking策略名称
        max_length: 对于greedy chunker的最大长度参数
    
    Returns:
        Chunker实例
    """
    if chunking_strategy == "sentence":
        return SentenceChunker()
    elif chunking_strategy == "paragraph":
        return ParagraphChunker()
    elif chunking_strategy == "greedy_sentence":
        return GreedySentenceChunker(max_length=max_length or 512)
    elif chunking_strategy == "greedy_paragraph":
        return GreedyParagraphChunker(max_length=max_length or 512)
    elif chunking_strategy == "token":
        return TokenChunker(config={"model_name": model_name, "max_length": max_length or 512, "overlap": overlap or 0})
    else:   
        raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")