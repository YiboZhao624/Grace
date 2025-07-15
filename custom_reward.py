import re


def format_reward(text):
    # check the format of the output text.
    # it should either contain <llm>...</llm> or <evidence>...</evidence> (but not both)
    # and contain <answer>...</answer> (independent, not nested)
    tag_count = 0
    tag_count += text.count("<llm>")
    tag_count += text.count("</llm>")
    tag_count += text.count("<evidence>")
    tag_count += text.count("</evidence>")
    tag_count += text.count("<answer>")
    tag_count += text.count("</answer>")
    
    if tag_count != 4:
        return 0
    
    llm_pattern = r"<llm>.*?</llm>"
    evidence_pattern = r"<evidence>.*?</evidence>"
    answer_pattern = r"<answer>.*?</answer>"
    
    has_llm_complete = re.search(llm_pattern, text, re.DOTALL) is not None
    has_evidence_complete = re.search(evidence_pattern, text, re.DOTALL) is not None
    has_answer_complete = re.search(answer_pattern, text, re.DOTALL) is not None
    
    if not has_answer_complete:
        return 0
    
    has_exactly_one_complete = (has_llm_complete and not has_evidence_complete) or (has_evidence_complete and not has_llm_complete)
    
    if not has_exactly_one_complete:
        return 0
    
    if has_llm_complete:
        llm_match = re.search(llm_pattern, text, re.DOTALL)
        if llm_match:
            llm_content = llm_match.group(0)
            if "<answer>" in llm_content:
                return 0
    
    if has_evidence_complete:
        evidence_match = re.search(evidence_pattern, text, re.DOTALL)
        if evidence_match:
            evidence_content = evidence_match.group(0)
            if "<answer>" in evidence_content:
                return 0
    
    return 0.5

def choice_reward(data):
    # if the model choose the correct tag, give 0.5 reward
    # else, give 0 reward.
    gt_choice = data["extra_info"]["gt_choice"]
    choice = data["output"].split(">")[0] + ">"
    if choice == gt_choice:
        return 0.5
    else:
        return 0

def evidence_reward(data):
    # calculate the evidence score, i.e.,
    # metric 1: How many ground truth evidence has been selected, i.e., recall.
    # metric 2: How precise the evidence is, i.e., precision.
    def cal_recall(gt_evidence, selected_evidence):
        pass

    def cal_precision(gt_evidence, selected_evidence):
        pass
    pass

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

def reward_function(data):
    total_reward = 0
    text = data["output"]
    total_reward += format_reward(text)

    return total_reward


