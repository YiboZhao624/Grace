from rouge_score import rouge_scorer


scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def rouge_l_reward(generated_text: str, ground_truth_list: list[str], reward_multiplier: float = 5.0):
    """ROUGE-L F1 as reward, and amplify the reward value to align with the <llm> part."""
    if not generated_text or not ground_truth_list:
        return 0.0

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
    if len(gt_evidence) == 1:
        if gt_evidence == [""]:
            gt_choice = "<llm>"
        else:
            gt_choice = "<evidence>"
    else:
        gt_choice = "evidence"

    choice = "<evidence>" if "<evidence>" in data else "<llm>"
    if choice == gt_choice and choice == "<evidence>":
        return 1, choice
    elif choice == gt_choice and choice == "<llm>":
        return 3, choice
    else:
        return 0, choice


def length_penalty(generated_text: str, ground_truth_list: list[str], min_length_ratio: float = 0.8):
    """
    if the length of the generated text is significantly shorter than the average length of the ground truth, then penalize the reward.
    """
    if not ground_truth_list:
        return 0.0
    
    avg_gt_len = sum(len(gt) for gt in ground_truth_list) / len(ground_truth_list)
    gen_len = len(generated_text)
    
    # if the length of the generated text is significantly shorter than the average length of the ground truth, then penalize the reward.
    if gen_len < avg_gt_len * min_length_ratio:
        return - (1.0 - gen_len / (avg_gt_len * min_length_ratio))
    
    return 0.0

def reward_function(data,gt_evidences:List[str], gt_answers:List[str])->dict:
    '''
    return a dictionary with key of score and reward_extra_info.
    score is the total reward, and the reward_extra_info is a dictionary with key of choice_r, format_r, evidence_r, answer_r.
    '''
    total_reward = 0
    text = data.split("</think>")[-1].strip()
    format_r = format_reward(text)
    if format_r == 0 :
        return {"score": 0, "choice": 0, "choice_r": 0, "evidence": 0, "answer": 0, "format": 0, "alp":0, "elp":0}
    total_reward += format_r

    choice_r, choice = choice_reward(text, gt_evidences)
    choice_report = 1 if choice == "<llm>" else 0
    if choice_r == 0:
        return {"score": total_reward, "choice": choice_report, "choice_r": choice_r, "evidence": 0, "answer": 0, "format": format_r, "alp":0, "elp":0}
    total_reward += choice_r
    answer_r = 0
    evidence_r = 0
    answer_text = text.split("</answer>")[0][len("<answer>"):]
    evidence_text = text.split("</evidence>")[0][len("<evidence>"):]

    if choice == "<evidence>":
        answer_r = max(answer_r, rouge_l_reward(answer_text, gt_answers, reward_multiplier=3.0))
        evidence_r = max(evidence_r, rouge_l_reward(evidence_text, gt_evidences, reward_multiplier=2.0))
        total_reward += evidence_r
        total_reward += answer_r
        answer_length_punish = length_penalty(answer_text, gt_answers)
        total_reward += answer_length_punish
        evidence_length_punish = length_penalty(evidence_text, gt_evidences)
        total_reward += evidence_length_punish
        return {"score": total_reward, "choice": choice_report, "choice_r": choice_r, "evidence": evidence_r, "answer": answer_r, "format": format_r, "alp": answer_length_punish, "elp": evidence_length_punish}
    elif choice == "<llm>":
        total_reward += answer_r
        return {"score": total_reward, "choice": choice_report, "choice_r": choice_r, "evidence": evidence_r, "answer": answer_r, "format": format_r, "alp":0, "elp":0}
    else:
        return {"score": 0, "choice": 0, "choice_r": 0, "evidence": 0, "answer": 0, "format": 0, "alp":0, "elp":0}

def val_reward_function(data,gt_evidences:List[str], gt_answers:List[str], gt_evidence_retrieved:bool)->dict:
    total_reward = 0
    text = data.split("</think>")[-1].strip()
    format_r = format_reward(text)
    if format_r == 0 :
        return {"score": 0, "choice": 0, "choice_r": 0, "evidence": 0, "answer": 0, "format": 0, "alp":0, "elp":0}
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
        choice = 0
        answer_r = max(answer_r, rouge_l_reward(answer_text, gt_answers, reward_multiplier=5.0))
        evidence_r = max(evidence_r, rouge_l_reward(evidence_text, gt_evidences, reward_multiplier=5.0))
        total_reward += evidence_r
        total_reward += answer_r
        alp = length_penalty(answer_text, gt_answers)
        total_reward += alp
        elp = length_penalty(evidence_text, gt_evidences)
        total_reward += elp
        return {"score": total_reward, "choice": choice, "choice_r": choice_r, "evidence": evidence_r, "answer": answer_r, "format": format_r, "alp": alp, "elp": elp}
    elif choice == "<llm>":
        choice = 1
        total_reward += answer_r
        return {"score": total_reward, "choice": choice, "choice_r": choice_r, "evidence": evidence_r, "answer": answer_r, "format": format_r, "alp": 0, "elp": 0}
    else:
        return {"score": 0, "choice": 0, "choice_r": 0, "evidence": 0, "answer": 0, "format": 0, "alp": 0, "elp": 0}

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
       if gt_evidence_retrieved is None:
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
    ans = '''<think>
Okay, let's tackle this question: "Who annotated the dataset?" The user provided three references, so I need to check each one for relevant information.

Starting with reference 1: It mentions the Event2Mind dataset, which is annotated for intent and reaction. The text says that events come from four sources and that they filtered out non-human activities. They also converted Event2Mind instances into first-person past-tense activities. But it doesn't explicitly say who did the annotation. However, since Event2Mind is a known dataset, maybe the annotation was done by the authors of that dataset. But the current document's authors might have used it, so maybe they are the ones who annotated the dataset here? Wait, the question is about the dataset mentioned in the current references. The current dataset being discussed here is the one built using Event2Mind and their own surveys. The text says they built their set of queries from Event2Mind and their own surveys. So the annotation of the Event2Mind dataset is done by its original creators, but the current dataset's queries are a combination of Event2Mind and their own. However, the question is about who annotated the dataset (the one in the current study). The references don't mention who annotated the dataset they created. They only mention using Event2Mind, which was annotated by others. So maybe the answer is that the dataset was annotated by the authors of the current study, but the references don't specify. Wait, the first reference says "the Event2Mind dataset BIBREF15" but doesn't mention annotation details. The current dataset's queries are built from Event2Mind and their own surveys. The annotation for the current dataset's activities might be done by the authors, but the references don't explicitly state that. 

Looking at reference 3: It talks about filtering tweets and extracting activities, but again, no mention of who annotated the dataset. 

So, the references don't directly state who annotated the dataset. The Event2Mind dataset's annotation is mentioned, but the current dataset's annotation isn't. Therefore, the answer should be that the references don't provide information on who annotated the dataset, and I need to recall that the Event2Mind dataset was annotated by its creators, but the current dataset's annotation isn't specified. Wait, but the question is about the dataset in the current study. The current study's dataset is built using Event2Mind and their own queries. The annotation for the current dataset's activities might be part of their process, but the references don't mention who did the annotation. Therefore, the answer would be that the references don't specify, and I need to use my knowledge. However, the user's instructions say if the evidence isn't relevant, I should recall. Since the references don't mention the annotators of the dataset in question, I should use my knowledge. But the dataset in question is the one built in the study, which uses Event2Mind and their own queries. The annotation of the Event2Mind dataset is done by its creators, but the current dataset's annotation isn't mentioned. Therefore, the answer would be that the dataset was annotated by the authors of the study, but the references don't specify. Wait, but the user wants the answer based on the evidence or my knowledge. Since the evidence doesn't mention, I should use my knowledge. However, the question is about the dataset mentioned in the references. The dataset in the references is the one built using Event2Mind and their own surveys. The annotation of the current dataset's activities isn't mentioned. Therefore, the answer is that the references don't provide information on who annotated the dataset, so I need to recall. But the user's instructions say if the evidence isn't relevant, I should recall. So the answer would be that the dataset was annotated by the authors of the study, but the references don't explicitly state this. Wait, but the first reference says they built their queries from Event2Mind and their own surveys. The annotation of the Event2Mind dataset is done by others, but the current dataset's queries are their own. However, the actual annotation of the tweets or activities in the current dataset isn't mentioned. Therefore, the answer is that the references don't specify who annotated the dataset, so I need to use my knowledge. But since the user is asking about the dataset in the references, which is the one built by the authors, perhaps the answer is that the dataset was annotated by the authors of the study. But the references don't explicitly say that. Hmm, this is tricky. The correct approach is to check the references again. 

Looking back, reference 1 mentions that the Event2Mind dataset is annotated for intent and reaction. The current study uses this dataset as a source for their queries. The current dataset's activities are derived from tweets, which are filtered and processed. However, the annotation of the tweets (i.e., labeling them as activities) isn't mentioned. The annotation of the Event2Mind dataset is done by others, but the current study's dataset isn't annotated by anyone mentioned in the references. Therefore, the answer should be that the references don't provide information on who annotated the dataset, so the answer would be based on my knowledge. But the user's instructions say to use the evidence if it's relevant. Since the evidence doesn't mention the annotators, I need to use my knowledge. However, the question is about the dataset in the references, which is the one built in the study. The study's authors would be the ones who annotated the dataset, but the references don't state that. Therefore, the answer is that the references don't specify, and the answer should be based on my knowledge. But the user wants the answer wrapped in <answer>...</answer>. Since the references don't mention the annotators, the correct process is to use my knowledge. But I need to confirm if the dataset's annotation is part of the study. The study's dataset is built by querying tweets, which are then processed. The annotation might be done by the authors, but the references don't say. Therefore, the answer is that the dataset was annotated by the authors of the study, but the references don't explicitly state this. However, since the user's instructions say to use evidence if possible, and the evidence doesn't mention it, I should use my knowledge. So the answer would be that the dataset was annotated by the authors of the study. But I'm not sure. Alternatively, maybe the Event2Mind dataset's annotation is the key here, but the question is about the dataset in the current study. Therefore, the answer is that the references don't provide information on who annotated the dataset, so the answer is based on my knowledge. But the user's instructions require that if the evidence isn't relevant, I should recall. Therefore, the answer would be that the dataset was annotated by the authors of the study, but the references don't specify. Wait, but the question is about the dataset mentioned in the references. The references don't mention the annotators, so the answer should be that the references don't provide that information. Therefore, I need to use my knowledge. However, the user might expect that the dataset in the references is the one built by the authors, so they would be the ones who annotated it. But the references don't explicitly say that. So the answer is that the dataset was annotated by the authors of the study, but the references don't specify. But the user's instructions say to use evidence if possible. Since the evidence doesn't mention, the answer should be based on my knowledge. So, the answer is that the dataset was annotated by the authors of the study, but the references don't explicitly state this. Therefore, the correct answer is that the references don't provide the information, so I need to use my knowledge. However, the user's example might expect that the answer is the authors. But I need to follow the process strictly. Since the evidence doesn't mention the annotators, the answer is based on my knowledge. But the question is about the dataset in the references, which is the one built using Event2Mind and their own surveys. The annotation of the tweets in the dataset would be done by the authors, but the references don't state that. Therefore, the answer is that the dataset was annotated by the authors of the study, but the references don't explicitly mention it. However, according to the instructions, if the evidence isn't relevant, I should recall. Since the evidence doesn't mention the annotators, I need to recall. So the answer would be that the dataset was annotated by the authors of the study. But I'm not entirely sure. Alternatively, maybe the answer is that the Event2Mind dataset was annotated by its creators, but the current dataset's annotation isn't mentioned. Therefore, the answer is that the references don't specify, so the answer is based on my knowledge. But I need to make a decision here. Given the instructions, since the evidence doesn't mention the annotators of the dataset in question, the answer should be based on my knowledge. So, the answer is that the dataset was annotated by the authors of the study. But I'm not certain. Alternatively, maybe the answer is that the dataset was annotated by the Event2Mind creators. But the question is about the dataset in the current study, not Event2Mind. Therefore, the answer is that the references don't provide the information, so the answer is based on my knowledge. But I have to choose. Given that the dataset in the current study is built using Event2Mind and their own queries, the annotation of the current dataset's activities would be done by the authors. So the answer is that the dataset was annotated by the authors of the study. However, the references don't explicitly state this, so according to the instructions, I should use my knowledge. Therefore, the answer is that the dataset was annotated by the authors of the study.
</think>

<evidence>
'''
    from utils import read_parquet
    path = "data/merged/0829-QASPER-soft_deduplicated-0.4.parquet"
    data = read_parquet(path)
    data = data[:5]
    print(len(data))
    for item in data:
        gt = item["reward_model"]["ground_truth"]
        reward = default_compute_score("QASPER", ans, gt)
        print(reward)
