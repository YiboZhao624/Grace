from rouge_score import rouge_scorer

# 在文件顶部全局初始化，避免重复创建对象
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def rouge_l_reward(generated_text: str, ground_truth_list: list[str], reward_multiplier: float = 5.0):
    """使用 ROUGE-L F1 计算奖励，并放大奖励值"""
    if not generated_text or not ground_truth_list:
        return 0.0

    # 计算与每个标准答案的分数，取最高值
    max_f1 = 0.0
    for gt in ground_truth_list:
        if not gt: continue
        scores = scorer.score(gt, generated_text)
        f1 = scores['rougeL'].fmeasure
        if f1 > max_f1:
            max_f1 = f1

    return max_f1 * reward_multiplier

import re
from typing import List

def format_reward(data:str):
    # check the format of the output text.
    # it should either contain <llm>...</llm> or <evidence>...</evidence> (but not both)
    # and contain <answer>...</answer> (independent, not nested)
    tag_count = 0
    tag_count += data.count("<llm>")
    tag_count += data.count("</llm>")
    tag_count += data.count("<evidence>")
    tag_count += data.count("</evidence>")
    tag_count += data.count("<answer>")
    tag_count += data.count("</answer>")
    
    if tag_count != 4:
        return 0
    
    llm_pattern = r"<llm>.*?</llm>"
    evidence_pattern = r"<evidence>.*?</evidence>"
    answer_pattern = r"<answer>.*?</answer>"
    
    has_llm_complete = re.search(llm_pattern, data, re.DOTALL) is not None
    has_evidence_complete = re.search(evidence_pattern, data, re.DOTALL) is not None
    has_answer_complete = re.search(answer_pattern, data, re.DOTALL) is not None
    
    if not has_answer_complete:
        return 0
    
    has_exactly_one_complete = (has_llm_complete and not has_evidence_complete) or (has_evidence_complete and not has_llm_complete)
    
    if not has_exactly_one_complete:
        return 0
    
    if has_llm_complete:
        llm_match = re.search(llm_pattern, data, re.DOTALL)
        if llm_match:
            llm_content = llm_match.group(0)
            if "<answer>" in llm_content:
                return 0
    
    if has_evidence_complete:
        evidence_match = re.search(evidence_pattern, data, re.DOTALL)
        if evidence_match:
            evidence_content = evidence_match.group(0)
            if "<answer>" in evidence_content:
                return 0
    
    return 0.5

def choice_reward(data:str, gt_evidence):
    # if the model choose the correct tag, give 0.5 reward
    # else, give 0 reward.
    gt_choice = "<evidence>" if gt_evidence != [""] else "<llm>"
    choice = "<evidence>" if "<evidence>" in data else "<llm>"
    if choice == gt_choice and choice == "<evidence>":
        return 1, choice
    elif choice == gt_choice and choice == "<llm>":
        return 5, choice
    else:
        return 0, choice


def bonus_reward(format_reward, choice_reward,
         evidence_reward, answer_reward):
    # check the alignment of the model output.
    # if the model choose the evidence tag correctly, return the correct evidence and answer with the designed format, return a large bonus.
    # if the model choose the llm tag correctly, return the answer aligned with the original model with designed format, return a large bonus.
    pass

def recitation_punish(format_reward, choice_reward,
         evidence_reward, answer_reward):
    # if the model choose the evidence tag correctly, but select the evidence incorrectly with a correct answer, return a larger penalty.
    pass

def reward_function(data,gt_evidences:List[str], gt_answers:List[str])->dict:
    '''
    return a dictionary with key of score and reward_extra_info.
    score is the total reward, and the reward_extra_info is a dictionary with key of choice_r, format_r, evidence_r, answer_r.
    '''
    total_reward = 0
    text = data
    format_r = format_reward(text)
    total_reward += format_r

    choice_r, choice = choice_reward(text, gt_evidences)
    choice_report = 1 if choice == "<llm>" else 0
    if choice_r == 0:
        return {"score": total_reward, "choice": choice, "choice_r": choice_r, "evidence": 0, "answer": 0, "format": format_r}
    total_reward += choice_r
    answer_r = 0
    evidence_r = 0
    answer_text = text.split("</answer>")[0][len("<answer>"):]
    evidence_text = text.split("</evidence>")[0][len("<evidence>"):]

    if choice == "<evidence>":
        answer_r = max(answer_r, rouge_l_reward(answer_text, gt_answers, reward_multiplier=5.0))
        evidence_r = max(evidence_r, rouge_l_reward(evidence_text, gt_evidences, reward_multiplier=4.0))
        total_reward += evidence_r
        total_reward += answer_r
        return {"score": total_reward, "choice": choice_report, "choice_r": choice_r, "evidence": evidence_r, "answer": answer_r, "format": format_r}
    elif choice == "<llm>":
        total_reward += answer_r
        return {"score": total_reward, "choice": choice_report, "choice_r": choice_r, "evidence": evidence_r, "answer": answer_r, "format": format_r}
    else:
        return {"score": 0, "choice": 0, "choice_r": 0, "evidence": 0, "answer": 0, "format": 0}

def val_reward_function(data,gt_evidences:List[str], gt_answers:List[str], gt_evidence_retrieved:bool)->dict:
    total_reward = 0
    text = data
    format_r = format_reward(text)
    total_reward += format_r

    if gt_evidence_retrieved:
        choice_r, choice = choice_reward(text, gt_evidences)
    else:
        choice_r, choice = choice_reward(text, [""])
    total_reward += choice_r
    answer_r = 0
    evidence_r = 0
    answer_text = text.split("</answer>")[0][len("<answer>"):]
    evidence_text = text.split("</evidence>")[0][len("<evidence>"):]
    if choice == "<evidence>":
        answer_r = max(answer_r, rouge_l_reward(answer_text, gt_answers, reward_multiplier=5.0))
        evidence_r = max(evidence_r, rouge_l_reward(evidence_text, gt_evidences, reward_multiplier=4.0))
        total_reward += evidence_r
        total_reward += answer_r
        return {"score": total_reward, "choice": choice_r, "choice_r": choice_r, "evidence": evidence_r, "answer": answer_r, "format": format_r}
    elif choice == "<llm>":
        total_reward += answer_r
        return {"score": total_reward, "choice": choice_r, "choice_r": choice_r, "evidence": evidence_r, "answer": answer_r, "format": format_r}
    else:
        return {"score": 0, "choice": 0, "choice_r": 0, "evidence": 0, "answer": 0, "format": 0}

def default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None, memory_limit_mb=None): 
    """Compute the score for a given solution based on the data source.
    This function is developed based on the veRL's default reward function. 
    The only difference is adding the custom reward function for QASPER.
    it will return a dictionary with the score and the extra information.
    To supervise the extra information, please refer to https://github.com/volcengine/verl/issues/2279 this issue.
    This function will be used to initialize the reward manager of DAPO at `verl/verl/workers/reward_manager/dapo.py`, line 94. It will return a dictionary and 
  
    Args: 
         data_source (str): The source dataset identifier which determines the scoring method. 
         solution_str (str): The solution string to be evaluated. 
         ground_truth (str): The ground truth answer for comparison. 
         extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None. 
  
     Returns: 
         float: The computed score as a floating point number. If the result is a dictionary, 
                it returns the dictionary instead. 
  
     Raises: 
         NotImplementedError: If the reward function is not implemented for the given data source. 
     """ 
    if data_source in ["QASPER"]:
       gt_evidence = ground_truth["gt_evidence"]
       gt_answer = ground_truth["answer"]
       gt_evidence_retrieved = ground_truth.get("gt_evidence_retrieved", None)
       if gt_evidence_retrieved is not None:
           res = reward_function(solution_str, gt_evidence, gt_answer)
       else:
           res = val_reward_function(solution_str, gt_evidence, gt_answer, gt_evidence_retrieved)
    else: 
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}") 
 
    if isinstance(res, dict): 
        return res 
    elif isinstance(res, (int, float, bool)): 
        return float(res) 
    else: 
        return float(res[0])


if __name__ == "__main__":
    data = "<llm>This is a test</llm><answer>This is a test</answer>"
    gt_evidence = [""]
    answer = choice_reward(data, gt_evidence)
    print(answer)