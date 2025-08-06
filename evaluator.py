from typing import List, Literal
from rouge_score import rouge_scorer
from bert_score import score as bert_score_compute
import torch
import re
import string
from tqdm import tqdm
import evaluate

def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)

def white_space_fix(text):
    return ' '.join(text.split())

def handle_punc(text):
    exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
    return ''.join(ch if ch not in exclude else ' ' for ch in text)

def lower(text):
    return text.lower()

def replace_underscore(text):
    return text.replace('_', ' ')


class Evaluator:
    METRICS_LITERAL = Literal["RL", "BL", "EM", "F1", "PR", "RE", "BS", "LJ", "RR"]
    def __init__(self, metrics: List[METRICS_LITERAL], **kwargs):
        if "LJ" in metrics:
            try:
                self.gpt = kwargs["LJ_api"]
            except KeyError:
                raise KeyError("LLM-as-a-Judge(LJ) is included in the metrics, but the LLM api(callable) is not provided.")
        if "RL" in metrics:
            try:
                self.Rouge_L_scorer = evaluate.load("rouge")
            except Exception as e:
                print(f"Error initializing Rouge-L scorer: {e}")
                raise 
        if "BL" in metrics:
            try:
                self.BLEU_scorer = evaluate.load("bleu")
            except Exception as e:
                print(f"Error initializing BLEU scorer: {e}")
                raise 
        if "F1" in metrics:
            try:
                self.F1_scorer = evaluate.load("f1")
            except Exception as e:
                print(f"Error initializing F1 scorer: {e}")
                raise 
        if "PR" in metrics:
            try:
                self.Precision_scorer = evaluate.load("precision")
            except Exception as e:
                print(f"Error initializing Precision scorer: {e}")
                raise 
        if "RE" in metrics:
            try:
                self.Recall_scorer = evaluate.load("recall")
            except Exception as e:
                print(f"Error initializing Recall scorer: {e}")
                raise 
        if "BS" in metrics:
            try:
                self.BERT_path = kwargs["BERT_path"]
            except KeyError:
                raise KeyError("BERT_path is not provided.")
            self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            print(f"Bert Model is loaded on the {self.device} for calculating BERT Score.")
        self.metrics = list(set(metrics))

    def _BLEU(self, answer: List[str], ground_truth: List[List[str]]) -> float:
        return self.BLEU_scorer.compute(predictions=answer, references=ground_truth)["bleu"]

    def _Exact_Match(self, answer: List[str], ground_truth: List[List[str]]) -> int:
        total = 0
        for i in range(len(answer)):
            for gt in ground_truth[i]:
                if white_space_fix(remove_articles(handle_punc(lower(replace_underscore(answer[i]))))).strip() == white_space_fix(remove_articles(handle_punc(lower(replace_underscore(gt))))).strip():
                    total += 1
                    break
        return total / len(answer)
    
    def _F1_score(self, answer: List[str], ground_truth: List[List[str]]) -> float:
        return self.F1_scorer.compute(predictions=[answer], references=[ground_truth])["f1"]

    def _Precision(self, answer: str, ground_truth: List[str]) -> float:
        return self.Precision_scorer.compute(predictions=[answer], references=[ground_truth])["precision"]

    def _Recall(self, answer: str, ground_truth: List[str]) -> float:
        return self.Recall_scorer.compute(predictions=[answer], references=[ground_truth])["recall"]
    
    def _LLM_Judge(self, answer: str, ground_truth: List[str]) -> int:
        LLM_Judge_prompt = '*************Consider a knowledge Q&A RAG task to test the capability of a testing model, the correct answer list is:*************\n' + "\n".join(ground_truth)
        LLM_Judge_prompt += '\n\n\n\n*************Here is the model\'s response:*************\n' + answer
        LLM_Judge_prompt += '\n\n\n\n*************Please check if the model\'s answer is correct. As long as the model\'s answer hits any item (or synonym) in the correct answer list, it can be considered correct. You only need to answer "yes" or "no".*************'
        return 1 if self.gpt.generate(LLM_Judge_prompt).lower() == "yes" else 0

    def evaluate(self, answers: List[str], ground_truths: List[List[str]]) -> dict:
        if len(answers) != len(ground_truths):
            raise ValueError("The number of answers and ground_truths must be the same.")
        num_samples = len(answers)
        results = [{} for _ in range(num_samples)]
        # --- Batch-Optimized Metrics ---
        if "RL" in self.metrics:
            rouge_results = self.Rouge_L_scorer.compute(
                predictions=answers,
                references=ground_truths,
                use_aggregator=False
            )
            for i, score in enumerate(rouge_results["rougeL"]):
                results[i]["Rouge-L-F1"] = score

        if "BS" in self.metrics:
            print("Computing BERTScore for the batch...")
            # BERTScore is inherently batch-friendly
            P, R, F1 = bert_score_compute(
                cands=answers,
                refs=ground_truths,
                model_type=self.BERT_path,
                lang="en",
                device=self.device,
                batch_size=64,
                verbose=True
            )
            for i in range(num_samples):
                results[i]["BERTScore-P"] = P[i].item()
                results[i]["BERTScore-R"] = R[i].item()
                results[i]["BERTScore-F1"] = F1[i].item()

        # --- Per-Sample Metrics ---
        print("Computing per-sample metrics (BLEU, EM, LLM-Judge)...")
        for i in tqdm(range(num_samples), desc="Processing samples"):
            if "BL" in self.metrics:
                results[i]["BLEU-4"] = self._BLEU(answers[i], ground_truths[i])
            if "EM" in self.metrics:
                results[i]["Exact Match"] = self._Exact_Match(answers[i], ground_truths[i])
            if "LJ" in self.metrics:
                results[i]["LLM-as-a-Judge"] = self._LLM_Judge(answers[i], ground_truths[i])

        return results

if __name__ == '__main__':
    # Mock API for LLM-as-a-Judge for demonstration purposes
    class MockLLMAPI:
        def call(self, prompt, api_key):
            return "yes"

    # Define which metrics to use
    enabled_metrics: List[Evaluator.METRICS_LITERAL] = ["RL", "BL", "EM", "BS", "LJ"]
    # Initialize the evaluator
    kwargs = {
        "LJ_api": MockLLMAPI(),
        "BERT_path": "./Roberta-large",
        "device": "cuda:0"
    }

    evaluator = Evaluator(metrics=enabled_metrics, **kwargs)

    # Prepare BATCH data (e.g., 3 samples)
    candidate_answers = [
        "The Eiffel Tower is located in Paris, France.",
        "The capital of Japan is Tokyo.",
        "The earth is flat."
    ]
    ground_truth_answers = [
        ["The Eiffel Tower is in Paris.", "Paris is the location of the Eiffel Tower."],
        ["Tokyo is the capital city of Japan."],
        ["The world is a sphere.", "The earth is round."]
    ]

    # Run the evaluation on the entire batch
    final_scores_batch = evaluator.evaluate(
        candidate_answers, 
        ground_truth_answers
    )

    # Print the results for each sample in the batch
    print("\n\n--- Batch Evaluation Results ---")
    for i, scores in enumerate(final_scores_batch):
        print(f"\n--- Sample {i+1}: '{candidate_answers[i]}' ---")
        for metric, score in scores.items():
            if isinstance(score, float):
                print(f"  {metric:<15}: {score:.4f}")
            else:
                print(f"  {metric:<15}: {score}")
            