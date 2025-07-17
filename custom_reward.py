import re

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
    gt_choice = "<evidence>" if gt_evidence else "<llm>"
    choice = data.split(">")[0] + ">"
    if choice == gt_choice:
        return 0.5, choice
    else:
        return 0, choice

def evidence_reward(data:str, gt_evidence:str):
    # calculate the evidence score, i.e.,
    # metric 1: How many ground truth evidence has been selected, i.e., recall.
    # metric 2: How precise the evidence is, i.e., precision.

    selected_evidence = data.split("</evidence>")[0].strip("<evidence>")
    # calculate the longest common substring between the selected evidence and the ground truth evidence
    def longest_common_substring(s1:str, s2:str):
        if not s1 or not s2:
            return 0
        m, n = len(s1), len(s2)
        prev = [0] * (n + 1)
        max_len = 0
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    curr[j] = prev[j - 1] + 1
                    max_len = max(max_len, curr[j])
            prev = curr
        return max_len
    
    lcs = longest_common_substring(gt_evidence, selected_evidence)
    precision = lcs / len(selected_evidence)
    recall = lcs / len(gt_evidence)

    return precision * recall

def answer_reward(data):
    # if the model return the correct answer, give 2 reward.
    # else, give 0 reward.
    pass

def bonus_reward(format_reward, choice_reward,
         evidence_reward, answer_reward):
    # check the alignment of the model output.
    # if the model choose the evidence tag correctly, return the correct evidence and answer with the designed format, return a large bonus.
    # if the model choose the llm tag correctly, return the answer aligned with the original model with designed format, return a large bonus.
    pass

def unalignment_punish(format_reward, choice_reward,
         evidence_reward, answer_reward):
    # check the alignment of the model output.
    # if the model choose the evidence tag incorrectly, but return the correct answer, return a penalty.
    # if the model choose the evidence tag correctly, but select the evidence incorrectly with a correct answer, return a larger penalty.
    pass

def reward_function(data,gt_evidence, gt_answer)->dict:
    '''
    return a dictionary with key of score and reward_extra_info.
    score is the total reward, and the reward_extra_info is a dictionary with key of choice_r, format_r, evidence_r, answer_r.
    '''
    total_reward = 0
    text = data["output"]
    format_r = format_reward(text)
    total_reward += format_r

    choice_r, choice = choice_reward(text, gt_evidence)
    total_reward += choice_r
    answer_r = answer_reward(text)
    evidence_r = evidence_reward(text, gt_evidence)

    if choice == "<evidence>":
        total_reward += evidence_r
    elif choice == "<llm>":
        pass
    else:
        return 0

    return {"score": total_reward, "extra_info": {"choice": choice, "choice_r": choice_r, "evidence": evidence_r, "answer": answer_r, "format": format_r}}


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
       gt_evidence = ground_truth["evidence"]
       gt_answer = ground_truth["answer"]
       res = reward_function(solution_str, gt_evidence, gt_answer)
    else: 
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}") 
 
    if isinstance(res, dict): 
        return res 
    elif isinstance(res, (int, float, bool)): 
        return float(res) 
    else: 
        return float(res[0])