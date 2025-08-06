from typing import List, Literal
from rouge_score import rouge_scorer
from bert_score import score as bert_score_compute
import torch
import re
import copy
import string
from tqdm import tqdm
import evaluate
from custom_reward import reward_function
from utils import setup_logging

logger = setup_logging("Evaluator")

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
                logger.error(f"Error initializing Rouge-L scorer: {e}")
                raise 
        if "BL" in metrics:
            try:
                self.BLEU_scorer = evaluate.load("bleu")
            except Exception as e:
                logger.error(f"Error initializing BLEU scorer: {e}")
                raise 
        if "F1" in metrics:
            try:
                self.F1_scorer = evaluate.load("f1")
            except Exception as e:
                logger.error(f"Error initializing F1 scorer: {e}")
                raise 
        if "PR" in metrics:
            try:
                self.Precision_scorer = evaluate.load("precision")
            except Exception as e:
                logger.error(f"Error initializing Precision scorer: {e}")
                raise 
        if "RE" in metrics:
            try:
                self.Recall_scorer = evaluate.load("recall")
            except Exception as e:
                logger.error(f"Error initializing Recall scorer: {e}")
                raise 
        if "BS" in metrics:
            try:
                self.BERT_path = kwargs["BERT_path"]
            except KeyError:
                raise KeyError("BERT_path is not provided.")
            self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Bert Model is loaded on the {self.device} for calculating BERT Score.")
        self.metrics = list(set(metrics))

    def _BLEU(self, answer: str, ground_truth:List[str]) -> float:
        return self.BLEU_scorer.compute(predictions=[answer], references=[ground_truth])["bleu"]

    def _Exact_Match(self, answer: str, ground_truth: List[str]) -> int:
        total = 0
        for gt in ground_truth:
            if white_space_fix(remove_articles(handle_punc(lower(replace_underscore(answer))))).strip() == white_space_fix(remove_articles(handle_punc(lower(replace_underscore(gt))))).strip():
                total += 1
                break
        return total
    
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

    def _RR(self, full_answer: str, ground_truth: List[str], ground_truth_evidences: List[str]) -> dict:
        return reward_function(full_answer, ground_truth, ground_truth_evidences)

    def evaluate(self,full_answer: List[str], chosen_evidences:List[str], answers: List[str], ground_truths: List[List[str]], ground_truth_evidences: List[List[str]]) -> dict:
        if len(answers) != len(ground_truths):
            raise ValueError("The number of answers and ground_truths must be the same.")
        logger.info(f"evaluating the answers and ground_truths with {self.metrics} metrics")
        num_samples = len(answers)
        evidence_results = {}
        answer_results = {}
        reward_results = {}
        # --- Batch-Optimized Metrics ---
        if "RL" in self.metrics:
            rouge_results = self.Rouge_L_scorer.compute(
                predictions=answers,
                references=ground_truths,
                use_aggregator=False
            )
            rouge_evidence_results = self.Rouge_L_scorer.compute(
                predictions=chosen_evidences,
                references=ground_truth_evidences,
                use_aggregator=False
            )
            for i, score in enumerate(rouge_results["rougeL"]):
                answer_results["Rouge-L-F1"] = answer_results.get("Rouge-L-F1", []) + [score]
            for i, score in enumerate(rouge_evidence_results["rougeL"]):
                evidence_results["Rouge-L-F1"] = evidence_results.get("Rouge-L-F1", []) + [score]

        if "BS" in self.metrics:
            logger.info("Computing BERTScore for the batch...")
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
            P_evidence, R_evidence, F1_evidence = bert_score_compute(
                cands=chosen_evidences,
                refs=ground_truth_evidences,
                model_type=self.BERT_path,
                lang="en",
                device=self.device,
                batch_size=64,
                verbose=True
            )
            for i in range(num_samples):
                answer_results["BERTScore-P"] = answer_results.get("BERTScore-P",[]) + [P[i].item()]
                answer_results["BERTScore-R"] = answer_results.get("BERTScore-R",[]) + [R[i].item()]
                answer_results["BERTScore-F1"] = answer_results.get("BERTScore-F1",[]) + [F1[i].item()]
            for i in range(num_samples):
                evidence_results["BERTScore-P"] = evidence_results.get("BERTScore-P",[]) + [P_evidence[i].item()]
                evidence_results["BERTScore-R"] = evidence_results.get("BERTScore-R",[]) + [R_evidence[i].item()]
                evidence_results["BERTScore-F1"] = evidence_results.get("BERTScore-F1",[]) + [F1_evidence[i].item()]

        # --- Per-Sample Metrics ---
        logger.info("Computing per-sample metrics (BLEU, EM, LLM-Judge)...")
        for i in tqdm(range(num_samples), desc="Processing samples"):
            if "BL" in self.metrics:
                answer_results["BLEU-4"] = answer_results.get("BLEU-4",[]) + [self._BLEU(answers[i], ground_truths[i])]
                evidence_results["BLEU-4"] = evidence_results.get("BLEU-4",[]) + [self._BLEU(chosen_evidences[i], ground_truth_evidences[i])]
            if "EM" in self.metrics:
                answer_results["Exact Match"] = answer_results.get("Exact Match",[]) + [self._Exact_Match(answers[i], ground_truths[i])]
                evidence_results["Exact Match"] = evidence_results.get("Exact Match",[]) + [self._Exact_Match(chosen_evidences[i], ground_truth_evidences[i])]
            if "LJ" in self.metrics:
                answer_results["LLM-as-a-Judge"] = answer_results.get("LLM-as-a-Judge",[]) + [self._LLM_Judge(answers[i], ground_truths[i])]
                # due to the evidence is directly copied, it is unnecessary to judge the evidence.
                # evidence_results[i]["LLM-as-a-Judge"] = self._LLM_Judge(chosen_evidences[i], ground_truth_evidences[i])
            if "RR" in self.metrics:
                result_RR = self._RR(full_answer[i], ground_truths[i], ground_truth_evidences[i])
                for key, value in result_RR.items():
                    reward_results[key] = reward_results.get(key, []) + [value]

        return answer_results, evidence_results, reward_results

if __name__ == '__main__':
    # Mock API for LLM-as-a-Judge for demonstration purposes
    class MockLLMAPI:
        def generate(self, prompt):
            return "yes"

    # Define which metrics to use
    enabled_metrics: List[Evaluator.METRICS_LITERAL] = ["RL", "BL", "EM", "BS", "LJ"]

    # Initialize the evaluator
    kwargs = {
        "LJ_api": MockLLMAPI(),
        "BERT_path": "bert-base-uncased",
        "device": "cuda:0"
    }

    evaluator = Evaluator(metrics=enabled_metrics, **kwargs)

    # Prepare BATCH data (e.g., 3 samples)
    candidate_answers = [
        "The Eiffel Tower is located in Paris, France.",
        "The capital of Japan is Tokyo.",
        "The earth is flat.",
        "The capital of China is Beijing.",
        "Jackie Chan"
    ]
    ground_truth_answers = [
        ["The Eiffel Tower is in Paris.", "Paris is the location of the Eiffel Tower."],
        ["Tokyo is the capital city of Japan."],
        ["The world is a sphere.", "The earth is round."],
        ["The capital of China is Beijing."],
        ["Green Table."]
    ]

    # Run the evaluation on the entire batch
    final_scores_batch = evaluator.evaluate(
        candidate_answers, 
        ground_truth_answers
    )

    # Print the results for each sample in the batch
    logger.info("\n\n--- Batch Evaluation Results ---")
    for i, scores in enumerate(final_scores_batch):
        logger.info(f"\n--- Sample {i+1}: '{candidate_answers[i]}' ---")
        for metric, score in scores.items():
            if isinstance(score, float):
                logger.info(f"  {metric:<15}: {score:.4f}")
            else:
                logger.info(f"  {metric:<15}: {score}")