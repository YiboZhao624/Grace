from typing import List, Literal, Union
import os
from rouge_score import rouge_scorer
from bert_score import score as bert_score_compute
import torch
import re
import copy
import string
from tqdm import tqdm
from custom_reward_v2 import val_reward_function
from utils import setup_logging, extract_evidence_or_none, extract_answer_or_all

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
        # Instance-local view
        self._metric_names = set(metrics)
        self.metrics = list(self._metric_names)

        # Optional LLM-as-a-Judge API
        if "LJ" in self._metric_names:
            try:
                self.gpt = kwargs["LJ_api"]
            except KeyError:
                raise KeyError("LLM-as-a-Judge(LJ) is included in the metrics, but the LLM api(callable) is not provided.")

        # BERTScore config (lazy model load happens inside bert_score_compute on first call)
        if "BS" in self._metric_names:
            try:
                self.BERT_path = kwargs["BERT_path"]
            except KeyError:
                raise KeyError("BERT_path is not provided.")
            self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"BERTScore will run on device: {self.device}")

            # Enable offline mode by default to avoid accidental network calls
            self.offline = kwargs.get("offline", True)
            self.skip_bs_on_error = kwargs.get("skip_bs_on_error", True)
            if self.offline:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
                os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            # Check if the provided path is a local directory (so we can safely use it offline)
            self._bert_path_is_local_dir = os.path.isdir(self.BERT_path)

        # Global ROUGE scorer (native library), instantiated once per process
        if not hasattr(Evaluator, "_ROUGE_SCORER"):
            Evaluator._ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # -------------------------
    # Native metric implementations
    # -------------------------
    def _get_tokens(self, text: str) -> List[str]:
        normalized = white_space_fix(
            remove_articles(handle_punc(lower(replace_underscore(text))))
        )
        if not normalized:
            return []
        return normalized.split()

    def _sentence_bleu(self, cand: str, refs: List[str], max_order: int = 4, smooth: bool = True) -> float:
        # Simple BLEU (up to 4-gram) with smoothing, single hypothesis vs multiple references
        cand_tokens = self._get_tokens(cand)
        refs_tokens = [self._get_tokens(r) for r in refs]
        if len(cand_tokens) == 0:
            return 0.0

        # Brevity penalty
        cand_len = len(cand_tokens)
        ref_lens = [len(rt) for rt in refs_tokens]
        if not ref_lens:
            return 0.0
        closest_ref_len = min(ref_lens, key=lambda rl: (abs(rl - cand_len), rl))
        if cand_len > 0:
            bp = 1.0 if cand_len > closest_ref_len else (pow(2.718281828, 1 - closest_ref_len / cand_len))
        else:
            bp = 0.0

        precisions = []
        for n in range(1, max_order + 1):
            # Count n-grams in candidate
            cand_ngrams = {}
            for i in range(0, max(0, cand_len - n + 1)):
                ng = tuple(cand_tokens[i:i + n])
                cand_ngrams[ng] = cand_ngrams.get(ng, 0) + 1

            # Max reference counts for n-grams
            max_ref_counts = {}
            for rt in refs_tokens:
                rt_len = len(rt)
                ref_counts = {}
                for i in range(0, max(0, rt_len - n + 1)):
                    ng = tuple(rt[i:i + n])
                    ref_counts[ng] = ref_counts.get(ng, 0) + 1
                for ng, cnt in ref_counts.items():
                    if cnt > max_ref_counts.get(ng, 0):
                        max_ref_counts[ng] = cnt

            overlap = 0
            total = 0
            for ng, cnt in cand_ngrams.items():
                overlap += min(cnt, max_ref_counts.get(ng, 0))
                total += cnt

            if total == 0:
                precision = 0.0
            else:
                if smooth:
                    precision = (overlap + 1) / (total + 1)
                else:
                    precision = overlap / total
            precisions.append(precision)

        # geometric mean of precisions
        from math import log, exp
        if any(p == 0 for p in precisions):
            geo_mean = 0.0
        else:
            geo_mean = exp(sum(log(p) for p in precisions) / max_order)
        return bp * geo_mean

    def _BLEU(self, answer: str, ground_truth:List[str]) -> float:
        return self._sentence_bleu(answer, ground_truth)

    def _Exact_Match(self, answer: str, ground_truth: List[str]) -> int:
        total = 0
        for gt in ground_truth:
            if white_space_fix(remove_articles(handle_punc(lower(replace_underscore(answer))))).strip() == white_space_fix(remove_articles(handle_punc(lower(replace_underscore(gt))))).strip():
                total += 1
                break
        return total
    
    def _F1_score(self, answer: List[str], ground_truth: List[List[str]]) -> float:
        # Multi-label F1: compute F1 between predicted labels and each reference label set; take the best.
        pred_set = set(answer)
        def f1_for_ref(ref_labels: List[str]) -> float:
            ref_set = set(ref_labels)
            tp = len(pred_set & ref_set)
            fp = len(pred_set - ref_set)
            fn = len(ref_set - pred_set)
            if tp == 0:
                return 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)
        return max((f1_for_ref(gt) for gt in ground_truth), default=0.0)

    def _Precision(self, answer: str, ground_truth: List[str]) -> float:
        # Token-level precision between answer and union of references
        tokens_pred = set(white_space_fix(remove_articles(handle_punc(lower(replace_underscore(answer))))).split())
        tokens_ref = set()
        for gt in ground_truth:
            tokens_ref |= set(white_space_fix(remove_articles(handle_punc(lower(replace_underscore(gt))))).split())
        if not tokens_pred:
            return 0.0
        tp = len(tokens_pred & tokens_ref)
        fp = len(tokens_pred - tokens_ref)
        denom = tp + fp
        return tp / denom if denom > 0 else 0.0

    def _Recall(self, answer: str, ground_truth: List[str]) -> float:
        # Token-level recall between answer and union of references
        tokens_pred = set(white_space_fix(remove_articles(handle_punc(lower(replace_underscore(answer))))).split())
        tokens_ref = set()
        for gt in ground_truth:
            tokens_ref |= set(white_space_fix(remove_articles(handle_punc(lower(replace_underscore(gt))))).split())
        if not tokens_ref:
            return 0.0
        tp = len(tokens_pred & tokens_ref)
        fn = len(tokens_ref - tokens_pred)
        denom = tp + fn
        return tp / denom if denom > 0 else 0.0
    
    def _LLM_Judge(self, answer: str, ground_truth: List[str], group_name: str) -> int:
        if group_name == "gt_retrieved_fail":
            ground_truth = ["I don't know.", "The provided evidence is not enough to answer the question."]
        LLM_Judge_prompt = '*************Consider a knowledge Q&A RAG task to test the capability of a testing model, the correct answer list is:*************\n' + "\n".join(ground_truth)
        LLM_Judge_prompt += '\n\n\n\n*************Here is the model\'s response:*************\n' + answer
        LLM_Judge_prompt += '\n\n\n\n*************Please check if the model\'s answer is correct. As long as the model\'s answer hits any item (or synonym) in the correct answer list, it can be considered correct. You only need to answer "yes" or "no".*************'
        return 1 if self.gpt.generate(LLM_Judge_prompt).lower() == "yes" else 0

    def _RR(self, full_answer: str, ground_truth_answers: List[str], ground_truth_evidences: List[str], gt_evidence_retrieved: bool) -> dict:
        # reward_function(data, gt_evidences, gt_answers)
        return val_reward_function(full_answer, ground_truth_evidences, ground_truth_answers, gt_evidence_retrieved)

    def _clean(self, candidate_gts: Union[List[str], List[List[str]]], answers: List[str]) -> List[str]:
        assert len(candidate_gts) == len(answers), "make sure the number of candidate answers and answers are the same"
        if isinstance(candidate_gts[0], str):
            candidate_gts = [[candidate_answer] for candidate_answer in candidate_gts]
        filtered_empty_entries = []
        non_empty_answer = []
        non_empty_gts = []
        for i in range(len(candidate_gts)):
            answer = answers[i].strip()
            candidate_answer_list =[candidate_answer.strip() for candidate_answer in candidate_gts[i] if candidate_answer.strip()]
            candidate_gts[i] = candidate_answer_list
            if len(candidate_answer_list) == 0 and len(answer) == 0:
                filtered_empty_entries.append(1)
            elif len(candidate_answer_list) == 0:
                filtered_empty_entries.append(0)
            elif len(answer) == 0:
                filtered_empty_entries.append(0)
            else:
                non_empty_answer.append(answer)
                non_empty_gts.append(candidate_answer_list)
        return non_empty_answer, non_empty_gts, filtered_empty_entries

    def _demonstrate(self, metric_name: str, scores: List[float], full_answers: List[str], chosen_evidences: List[str], answers: List[str], ground_truths: List[List[str]], ground_truth_evidences: List[List[str]], num_examples: int = 3):
        """通过 logger 展示指定 metric 的 top N 和 bottom N 案例"""
        if not scores:
            return
            
        # 将分数和索引配对，以便排序后能找回原文
        indexed_scores = list(enumerate(scores))
        
        # 按分数排序
        sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        
        logger.info(f"\n{'='*20} Demonstrator for Metric: {metric_name} {'='*20}")
        
        # 展示 Top N
        logger.info(f"\n--- Top {num_examples} Highest Scores ---")
        for i in range(min(num_examples, len(sorted_scores))):
            original_index, score = sorted_scores[i]
            logger.info(f"  Rank {i+1}: Score = {score:.4f} (Original Index: {original_index})")
            logger.info(f"    - Generated Answer: {answers[original_index]}")
            logger.info(f"    - Ground Truths: {ground_truths[original_index]}")
            if chosen_evidences[original_index]: # 仅在有选择证据时展示
                logger.info(f"    - Chosen Evidence: {chosen_evidences[original_index]}")
                logger.info(f"    - GT Evidences: {ground_truth_evidences[original_index]}")
            logger.info("-" * 10)

        # 展示 Bottom N
        logger.info(f"\n--- Bottom {num_examples} Lowest Scores ---")
        for i in range(min(num_examples, len(sorted_scores))):
            original_index, score = sorted_scores[-(i+1)]
            logger.info(f"  Rank {len(sorted_scores)-i}: Score = {score:.4f} (Original Index: {original_index})")
            logger.info(f"    - Generated Answer: {answers[original_index]}")
            logger.info(f"    - Ground Truths: {ground_truths[original_index]}")
            if chosen_evidences[original_index]: # 仅在有选择证据时展示
                logger.info(f"    - Chosen Evidence: {chosen_evidences[original_index]}")
                logger.info(f"    - GT Evidences: {ground_truth_evidences[original_index]}")
            logger.info("-" * 10)
        
        logger.info(f"{'='*20} End of Demonstrator for {metric_name} {'='*20}\n")

    def _evaluate_group(self, entries: List[dict], group_name: str, num_examples: int = 3):
        """evaluate the entries in a group"""
        if not entries:
            return {}, {}, {}

        # 1. extract the data needed for evaluation from the entry list
        full_answers = [e["answer"] for e in entries]
        chosen_evidences = [extract_evidence_or_none(e["answer"]) for e in entries]
        answers = [extract_answer_or_all(e["answer"]) for e in entries]
        
        ground_truths = []
        ground_truth_evidences = []

        for e in entries:
            gt_data = e["reward_model"]["ground_truth"]
            ground_truths.append(gt_data["answer"])
            ground_truth_evidences.append(gt_data.get("gt_evidence", [""]))
        
        if len(answers) != len(ground_truths):
            raise ValueError("The number of answers and ground_truths must be the same.")
        logger.info(f"evaluating the answers and ground_truths with {self.metrics} metrics")
        num_samples = len(answers)
        evidence_results = {}
        answer_results = {}
        reward_results = {}
        # --- Batch-Optimized Metrics ---
        if "RL" in self.metrics:
            # Compute per-sample best Rouge-L-F1 across multiple references
            for i in range(num_samples):
                cand_ans = answers[i]
                refs_ans = ground_truths[i]
                best_rl_ans = 0.0
                for ref in refs_ans:
                    rl = Evaluator._ROUGE_SCORER.score(ref, cand_ans)["rougeL"].fmeasure
                    if rl > best_rl_ans:
                        best_rl_ans = rl
                answer_results["Rouge-L-F1"] = answer_results.get("Rouge-L-F1", []) + [best_rl_ans]

                cand_evd = chosen_evidences[i]
                refs_evd = ground_truth_evidences[i]
                best_rl_evd = 0.0
                for ref in refs_evd:
                    rl = Evaluator._ROUGE_SCORER.score(ref, cand_evd)["rougeL"].fmeasure
                    if rl > best_rl_evd:
                        best_rl_evd = rl
                evidence_results["Rouge-L-F1"] = evidence_results.get("Rouge-L-F1", []) + [best_rl_evd]

        if "BS" in self.metrics:
            logger.info("Computing BERTScore for the batch...")
            # BERTScore is inherently batch-friendly
            non_empty_answer, non_empty_gts, filtered_empty_entries = self._clean(ground_truths, answers)
            non_empty_evd, non_empty_evd_refs, filtered_empty_entries_evd = self._clean(ground_truth_evidences, chosen_evidences)
            logger.info(f"the number of non-empty answer is: {len(non_empty_answer)}")
            logger.info(f"the number of non-empty gts is: {len(non_empty_gts)}")
            logger.info(f"the number of non-empty evd is: {len(non_empty_evd)}")
            logger.info(f"the number of non-empty evd_refs is: {len(non_empty_evd_refs)}")
            logger.info(f"the number of filtered_empty_entries is: {len(filtered_empty_entries)}")
            logger.info(f"the number of filtered_empty_entries_evd is: {len(filtered_empty_entries_evd)}")
            try:
                P, R, F1 = bert_score_compute(
                    cands=non_empty_answer,
                    refs=non_empty_gts,
                    model_type=self.BERT_path,
                    lang="en",
                    device=self.device,
                    batch_size=64,
                    verbose=True,
                    use_fast_tokenizer=False
                )
            except TypeError:
                P, R, F1 = bert_score_compute(
                    cands=non_empty_answer,
                    refs=non_empty_gts,
                    model_type=self.BERT_path,
                    lang="en",
                    device=self.device,
                    batch_size=64,
                    verbose=True,
                )

            if len(non_empty_evd) > 0:
                try:
                    P_evidence, R_evidence, F1_evidence = bert_score_compute(
                        cands=non_empty_evd,
                        refs=non_empty_evd_refs,
                        model_type=self.BERT_path,
                        lang="en",
                        device=self.device,
                        batch_size=64,
                        verbose=True,
                        use_fast_tokenizer=False
                    )
                except TypeError:
                    P_evidence, R_evidence, F1_evidence = bert_score_compute(
                        cands=non_empty_evd,
                        refs=non_empty_evd_refs,
                        model_type=self.BERT_path,
                        lang="en",
                        device=self.device,
                        batch_size=64,
                        verbose=True
                    )
            else:
                P_evidence, R_evidence, F1_evidence = [], [], []
            for i in range(len(non_empty_answer)):
                answer_results["BERTScore-P"] = answer_results.get("BERTScore-P",[]) + [P[i].item()]
                answer_results["BERTScore-R"] = answer_results.get("BERTScore-R",[]) + [R[i].item()]
                answer_results["BERTScore-F1"] = answer_results.get("BERTScore-F1",[]) + [F1[i].item()]
            
            for i in range(len(non_empty_evd)):
                evidence_results["BERTScore-P"] = evidence_results.get("BERTScore-P",[]) + [P_evidence[i].item()]
                evidence_results["BERTScore-R"] = evidence_results.get("BERTScore-R",[]) + [R_evidence[i].item()]
                evidence_results["BERTScore-F1"] = evidence_results.get("BERTScore-F1",[]) + [F1_evidence[i].item()]

            evidence_results["BERTScore-P"] = evidence_results.get("BERTScore-P",[]) + filtered_empty_entries_evd
            evidence_results["BERTScore-R"] = evidence_results.get("BERTScore-R",[]) + filtered_empty_entries_evd
            evidence_results["BERTScore-F1"] = evidence_results.get("BERTScore-F1",[]) + filtered_empty_entries_evd

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
                answer_results["LLM-as-a-Judge"] = answer_results.get("LLM-as-a-Judge",[]) + [self._LLM_Judge(answers[i], ground_truths[i], group_name)]
                # due to the evidence is directly copied, it is unnecessary to judge the evidence.
                # evidence_results[i]["LLM-as-a-Judge"] = self._LLM_Judge(chosen_evidences[i], ground_truth_evidences[i], group_name)
            if "RR" in self.metrics:
                result_RR = self._RR(full_answers[i], ground_truths[i], ground_truth_evidences[i], group_name=="gt_retrieved_success")
                for key, value in result_RR.items():
                    reward_results[key] = reward_results.get(key, []) + [value]

        if num_examples > 0:
            logger.info(f"Demonstrating Top/Bottom {num_examples} entries for each metric...")
            all_results_for_demo = {**answer_results, **evidence_results, **reward_results}
            for metric_name, scores in all_results_for_demo.items():
                try:
                    self._demonstrate(
                        metric_name=metric_name,
                        scores=scores,
                        full_answers=full_answers,
                        chosen_evidences=chosen_evidences,
                        answers=answers,
                        ground_truths=ground_truths,
                        ground_truth_evidences=ground_truth_evidences,
                        num_examples=num_examples
                    )
                except Exception as e:
                    logger.error(f"Error demonstrating {metric_name}: {e}")
                    continue

        return answer_results, evidence_results, reward_results


    def evaluate(self, entries: List[dict], num_examples: int = 0) -> dict:
        logger.info(f"start evaluating {len(entries)} samples...")
        
        # 1. group the entries by the gt_evidence_retrieved flag
        groups = {
            "gt_retrieved_success": [],
            "gt_retrieved_fail": [],
        }
        for entry in entries:
            gt_info = entry.get("reward_model", {}).get("ground_truth", {})
            retrieved_flag = gt_info.get("gt_evidence_retrieved") # maybe True, False for test and eval data.

            if retrieved_flag is True:
                groups["gt_retrieved_success"].append(entry)
            elif retrieved_flag is False:
                groups["gt_retrieved_fail"].append(entry)
            else:
                logger.error(f"the gt_evidence_retrieved flag is {retrieved_flag}. It looks like you are evaluating the training data. Please check your file path.")
                raise
        
        # 2. evaluate the entries in each group
        all_results = {}
        for group_name, group_entries in groups.items():
            if group_entries:
                logger.info(f"evaluating the group: '{group_name}' ({len(group_entries)} samples)")
                answer_res, evidence_res, reward_res = self._evaluate_group(group_entries, group_name, num_examples)
                all_results[group_name] = {
                    "count": len(group_entries),
                    "answer_results": answer_res,
                    "evidence_results": evidence_res,
                    "reward_results": reward_res
                }

        return all_results


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