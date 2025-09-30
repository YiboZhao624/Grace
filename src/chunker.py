# This file is used to define the chunker class, which is used to chunk the documents into smaller chunks.
# If you want to modify the chunker, just inherit the Chunker class and override the methods.
from typing import List, Dict
from transformers import AutoTokenizer
from configs import ChunkerConfig

class Chunker:
    def __init__(self):
        pass

    def chunk(self, document: str) -> List[str]:
        raise NotImplementedError("Subclasses must implement this method")

class SentenceChunker(Chunker):
    def __init__(self):
        super().__init__()
        
    def chunk(self, document: str) -> List[str]:
        chunks = document.split(".")
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks

class ParagraphChunker(Chunker):
    def __init__(self):
        super().__init__()
        
    def chunk(self, document: str) -> List[str]:
        chunks = document.split("\n")
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks

class GreedySentenceChunker(Chunker):
    def __init__(self, max_length: int):
        super().__init__()
        self.max_length = max_length

    def chunk(self, document: str) -> List[str]:
        sentences = document.split(".")
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(sentence) > self.max_length - len(current_chunk):
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += sentence
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

class GreedyParagraphChunker(Chunker):
    def __init__(self, max_length: int):
        super().__init__()
        self.max_length = max_length

    def chunk(self, document: str) -> List[str]:
        paragraphs = document.split("\n")
        chunks = []
        current_chunk = ""
        for paragraph in paragraphs:
            if len(paragraph) > self.max_length - len(current_chunk):
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                current_chunk += paragraph
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

class TokenChunker(Chunker):
    def __init__(self, config: ChunkerConfig):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.length = config.max_length
        self.overlap = config.overlap

    def chunk(self, document: str) -> List[str]:
        token_ids = self.tokenizer.encode(document, add_special_tokens=False)
        chunks = []
        for i in range(0, len(token_ids), self.length - self.overlap):
            chunk_token_ids = token_ids[i:i+self.length]
            chunks.append(chunk_token_ids)
        chunks = [self.tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
        return chunks

def get_chunker(config: ChunkerConfig) -> Chunker:
    """
    根据策略名称获取chunker实例
    
    Args:
        chunking_strategy: chunking策略名称
        max_length: 对于greedy chunker的最大长度参数
    
    Returns:
        Chunker实例
    """
    chunking_strategy = config.chunking_strategy
    max_length = config.max_length
    overlap = config.overlap
    if chunking_strategy == "sentence":
        return SentenceChunker()
    elif chunking_strategy == "paragraph":
        return ParagraphChunker()
    elif chunking_strategy == "greedy_sentence":
        return GreedySentenceChunker(max_length=max_length or 512)
    elif chunking_strategy == "greedy_paragraph":
        return GreedyParagraphChunker(max_length=max_length or 512)
    elif chunking_strategy == "token":
        return TokenChunker(config)
    else:   
        raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")

if __name__ == "__main__":
    print("Now testing the chunker...")
    chunker = SentenceChunker()
    with open("test/test_doc.txt", "r") as f:
        document = f.read()
    chunks = chunker.chunk(document)
    assert len(chunks) == 20, "The number of chunks should be 20"
    print("SentenceChunker passed")

    chunker = ParagraphChunker()
    with open("test/test_doc.txt", "r") as f:
        document = f.read()
    chunks = chunker.chunk(document)
    assert len(chunks) == 12, "The number of chunks should be 12"
    print("ParagraphChunker passed")

    chunker = GreedySentenceChunker(max_length=100)
    with open("test/test_doc.txt", "r") as f:
        document = f.read()
    chunks = chunker.chunk(document)
    assert len(chunks) == 8, "The number of chunks should be 8"
    print("GreedySentenceChunker passed")
    
    chunker = GreedyParagraphChunker(max_length=100)
    with open("test/test_doc.txt", "r") as f:
        document = f.read()
    chunks = chunker.chunk(document)
    assert len(chunks) == 8, "The number of chunks should be 8"
    print("GreedyParagraphChunker passed")
    
    