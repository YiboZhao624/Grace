from rouge_score import rouge_scorer


scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def rouge_l_reward(generated_text: str, ground_truth_list: list[str], reward_multiplier: float = 5.0):
    """ROUGE-L F1 as reward, and amplify the reward value to align with the <llm> part."""
    if not generated_text or not any(ground_truth_list):
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
    if len(gt_evidence) == 0:
        gt_choice = "<llm>"
    elif len(gt_evidence) == 1:
        if gt_evidence == [""]:
            gt_choice = "<llm>"
        else:
            gt_choice = "<evidence>"
    else:
        gt_choice = "<evidence>"

    choice = "<evidence>" if "<evidence>" in data else "<llm>"
    if choice == gt_choice and choice == "<evidence>":
        return 1, choice
    elif choice == gt_choice and choice == "<llm>":
        return 1, choice
    else:
        return 0, choice


def length_penalty(generated_text: str, ground_truth_list: list[str], min_length_ratio: float = 0.8):
    """
    if the length of the generated text is significantly shorter than the average length of the ground truth, then penalize the reward.
    """
    if not any(ground_truth_list):
        return 0.0
    
    avg_gt_len = sum(len(gt) for gt in ground_truth_list) / len(ground_truth_list)
    gen_len = len(generated_text)
    
    # if the length of the generated text is significantly shorter than the average length of the ground truth, then penalize the reward.
    # add the constant 100 to avoid the evidence is extremely long.
    if gen_len < min(avg_gt_len * min_length_ratio, 100):
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
    answer_text = text.split("</answer>")[0].split("<answer>")[1]
    evidence_text = text.split("</evidence>")[0].split("<evidence>")[1]

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
    if choice_r == 0:
        choice = 0 if choice == "<evidence>" else 1
        return {"score": total_reward, "choice": choice, "choice_r": choice_r, "evidence": 0, "answer": 0, "format": format_r, "alp":0, "elp":0}
    total_reward += choice_r
    answer_r = 0
    evidence_r = 0
    answer_text = text.split("</answer>")[0].split("<answer>")[1]
    evidence_text = text.split("</evidence>")[0].split("<evidence>")[1]
    if choice == "<evidence>":
        choice = 0
        answer_r = max(answer_r, rouge_l_reward(answer_text, gt_answers, reward_multiplier=1.0))
        evidence_r = max(evidence_r, rouge_l_reward(evidence_text, gt_evidences, reward_multiplier=1.0))
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
    if data_source.lower() in ["qasper"]:
       gt_evidence = ground_truth["gt_evidence"]
       gt_answer = ground_truth["answer"]
       gt_evidence_retrieved = ground_truth.get("gt_evidence_retrieved", None)
       if gt_evidence_retrieved is None:
           res = reward_function(solution_str, gt_evidence, gt_answer)
       else:
           res = val_reward_function(solution_str, gt_evidence, gt_answer, gt_evidence_retrieved)
    elif data_source in ["HOTPOTQA", "2WIKIMULTIHOP"]:
        gt_evidence = " ".join(ground_truth["gt_evidence"])
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
Okay, let's tackle this question. The user is asking, "who annotated the dataset?" I need to check the provided references to find the answer.

Looking at the first reference, it mentions the Event2Mind dataset, which is annotated for intent and reaction. The text says, "The Event2Mind dataset contains a large number of event phrases which are annotated for intent and reaction." So, the Event2Mind dataset itself was annotated. But the question is about the dataset used in the study, which might be the one they built using Event2Mind and their own queries. 

Wait, the first paragraph of the first reference says they built their set of human activity queries from two sources: Event2Mind and their own surveys. Then, they mention that the Event2Mind dataset is annotated. However, the main dataset they're talking about in the study might be the one they created by using those queries. But the question is about who annotated the dataset. 

In the first reference, the Event2Mind dataset is mentioned as a source, and it's stated that it's annotated. But the study's own dataset might not be annotated by the authors. However, the text doesn't explicitly say who annotated the Event2Mind dataset. The user's question is about the dataset in the study, which could be the one built using Event2Mind and their own queries. But the answer might be that the Event2Mind dataset was annotated, and the study's dataset might not have annotations. 

Wait, the first reference says, "We build our set of human activity queries from two sources: the Event2Mind dataset BIBREF15 and a set of short activity surveys, which we collect ourselves, to obtain nearly 30K queries." So the Event2Mind dataset is a source, but the study's own dataset isn't mentioned as being annotated. The question is about the dataset they annotated. However, the text doesn't explicitly state who annotated the Event2Mind dataset. The answer might be that the Event2Mind dataset was annotated by its creators, but the study's dataset isn't annotated. However, the user's question is about the dataset in the study. 

Wait, the first paragraph says that they used Event2Mind and their own surveys to create queries. The Event2Mind dataset is annotated, but the study's dataset might not be. The question is about who annotated the dataset. If the dataset in question is the one they built using the queries, then maybe they didn't annotate it. However, the text doesn't mention annotations for their dataset. But the first reference does mention that the Event2Mind dataset is annotated. However, the user's question might be referring to the dataset they created, which might not have annotations. 

Alternatively, maybe the answer is that the Event2Mind dataset was annotated by its creators, and the study's dataset isn't annotated. But the question is about the dataset in the study. Wait, the first reference says that the Event2Mind dataset is annotated, but the study's dataset might not be. However, the user's question is about the dataset that was annotated. If the dataset in the study is the one built using Event2Mind and their own queries, then the annotations might come from Event2Mind. But the study's own dataset might not have annotations. 

Hmm, the answer might be that the Event2Mind dataset was annotated, but the study's dataset isn't mentioned as being annotated. However, the user's question is about the dataset in the study. Since the first reference mentions that the Event2Mind dataset is annotated, but the study's dataset isn't, perhaps the answer is that the Event2Mind dataset was annotated by its creators, but the study's dataset isn't. However, the user's question might be referring to the dataset in the study, which isn't explicitly mentioned as being annotated. 

Wait, the first reference says that the Event2Mind dataset is annotated for intent and reaction. The study's dataset might not be annotated. So the answer would be that the Event2Mind dataset was annotated, but the study's dataset isn't. However, the user's question is about the dataset they are referring to. Since the study's dataset isn't mentioned as being annotated, maybe the answer is that the Event2Mind dataset was annotated, and the study's dataset isn't. But the user's question is about the dataset in the study. 

Alternatively, perhaps the answer is that the dataset was annotated by the authors. Wait, the first reference says that the Event2Mind dataset is annotated, but the study's dataset isn't. So the answer is that the Event2Mind dataset was annotated by its creators, and the study's dataset isn't. However, the user's question is about the dataset in the study. Since the study's dataset isn't mentioned as being annotated, the answer might be that the Event2Mind dataset was annotated, but the study's dataset isn't. 

But the user's question is about the dataset in the study. The answer would be that the dataset was annotated by the authors. Wait, the first reference says that the Event2Mind dataset is annotated, but the study's dataset might not be. The text doesn't mention annotations for the study's dataset. So the answer is that the Event2Mind dataset was annotated, but the study's dataset isn't. Therefore, the answer is that the Event2Mind dataset was annotated by its creators, but the study's dataset isn't. However, the user's question is about the dataset in the study. 

Alternatively, maybe the answer is that the authors annotated the dataset. But the text doesn't say that. The first reference mentions that the Event2Mind dataset is annotated, but the study's dataset isn't. So the answer is that the Event2Mind dataset was annotated, but the study's dataset isn't. Therefore, the answer is that the Event2Mind dataset was annotated by its creators, and the study's dataset isn't. But the user's question is about the dataset in the study. 

Hmm, I think the answer is that the Event2Mind dataset was annotated, but the study's dataset isn't. However, the user's question might be referring to the dataset in the study, which isn't annotated. Therefore, the answer is that the dataset was not annotated, but the Event2Mind dataset was. However, the user's question is about the dataset in the study. 

Wait, the first reference says that the Event2Mind dataset is annotated, and the study's dataset is built from that. So the study's dataset might not have annotations. Therefore, the answer is that the Event2Mind dataset was annotated, but the study's dataset isn't. However, the user's question is about the dataset in the study. 

I think the answer is that the Event2Mind dataset was annotated, but the study's dataset isn't. Therefore, the answer is that the dataset was not annotated by the authors. But the user's question is about the dataset in the study. Since the study's dataset isn't mentioned as being annotated, the answer is that the dataset was not annotated. But the first reference mentions that the Event2Mind dataset is annotated. So the answer is that the Event2Mind dataset was annotated by its creators, and the study's dataset isn't. 

But the user's question is about the dataset in the study. Therefore, the answer is that the dataset was not annotated. However, the first reference says that the Event2Mind dataset is annotated, but the study's dataset isn't. Therefore, the answer is that the dataset was not annotated. But the user's question might be referring to the Event2Mind dataset. 

I'm a bit confused. Let me check again. The user's question is about the dataset. The first reference mentions that the Event2Mind dataset is annotated. The study's dataset is built using queries from Event2Mind and their own surveys. The study's dataset might not have annotations. Therefore, the answer is that the Event2Mind dataset was annotated, but the study's dataset isn't. However, the user's question is about the dataset in the study. So the answer is that the dataset was not annotated. 

But the user's question is about the dataset. If the dataset in question is the one built by the study, then the answer is that it wasn't annotated. But the first reference mentions that the Event2Mind dataset is annotated. So the answer is that the Event2Mind dataset was annotated by its creators, and the study's dataset isn't. Therefore, the answer is that the dataset (Event2Mind) was annotated by its creators. But the user's question is about the dataset in the study. 

Wait, the first reference says that the Event2Mind dataset is annotated, and the study's dataset is built from that. The study's dataset isn't mentioned as being annotated. Therefore, the answer is that the Event2Mind dataset was annotated, but the study's dataset isn't. Therefore, the answer to the user's question is that the dataset was not annotated. 

But the user's question is about the dataset in the study. Therefore, the answer is that the dataset was not annotated. However, the first reference says that the Event2Mind dataset is annotated. So the answer is that the Event2Mind dataset was annotated by its creators, but the study's dataset isn't. Therefore, the answer is that the dataset (Event2Mind) was annotated by its creators. 

But the user's question is about the dataset in the study. Therefore, the answer is that the dataset was not annotated. 

Hmm, I think the answer is that the Event2Mind dataset was annotated, but the study's dataset isn't. Therefore, the answer is that the dataset was not annotated. However, the user's question might be referring to the Event2Mind dataset. 

Alternatively, the answer is that the dataset was annotated by the authors. But the text doesn't mention that
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
