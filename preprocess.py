import os
import json
import pandas as pd
import numpy as np
import random

random.seed(42)

def process_qasper_data(data:dict):
    processed_QA_data = []
    processed_paper_data = {}
    for paper_key, value in data.items():
        paper_title = value["title"]
        abstract = value["abstract"]
        full_text = value["full_text"]
        if paper_key not in processed_paper_data.keys():
            processed_paper_data[paper_key] = {"title": paper_title, "abstract": abstract, "full_text": full_text}
        
        qa_infos = value["qas"]
        for qa_info in qa_infos:
            processed_one_entry_QA = {}
            processed_one_entry_QA["question_id"] = qa_info["question_id"]
            processed_one_entry_QA["paper_id"] = paper_key
            processed_one_entry_QA["question"] = qa_info["question"]
            references = []
            for annotation_info in qa_info["answers"]:
                answer_info = annotation_info["answer"]
                if answer_info["unanswerable"]:
                    references.append({"answer": "Unanswerable", "evidence": [], "type": "none"})
                else:
                    if answer_info["extractive_spans"]:
                        answer = ", ".join(answer_info["extractive_spans"])
                        answer_type = "extractive"
                    elif answer_info["free_form_answer"]:
                        answer = answer_info["free_form_answer"]
                        answer_type = "abstractive"
                    elif answer_info["yes_no"]:
                        answer = "Yes"
                        answer_type = "boolean"
                    elif answer_info["yes_no"] is not None:
                        answer = "No"
                        answer_type = "boolean"
                    else:
                        raise RuntimeError(f"Annotation {answer_info['annotation_id']} does not contain an answer")
                    evidence = answer_info["evidence"]
                    references.append({"answer": answer, "evidence": evidence, "type": answer_type})
            processed_one_entry_QA["references"] = references
            processed_QA_data.append(processed_one_entry_QA)
    return processed_QA_data, processed_paper_data

def preprocess_QASPER(data_folder):
    train_path = os.path.join(data_folder, "qasper-train-v0.3.json")
    val_path = os.path.join(data_folder, "qasper-dev-v0.3.json")
    test_path = os.path.join(data_folder, "qasper-test-v0.3.json")

    with open(train_path, "r") as f:
        train_data = json.load(f)

    with open(val_path, "r") as f:
        val_data = json.load(f)

    with open(test_path, "r") as f:
        test_data = json.load(f)

    train_QA_data, train_paper_data = process_qasper_data(train_data)
    val_QA_data, val_paper_data = process_qasper_data(val_data)
    test_QA_data, test_paper_data = process_qasper_data(test_data)
    
    print("train QA data: ", len(train_QA_data))
    print("train paper data: ", len(train_paper_data))
    
    print("val QA data: ", len(val_QA_data))
    print("val paper data: ", len(val_paper_data))
    
    print("test QA data: ", len(test_QA_data))
    print("test paper data: ", len(test_paper_data))

    os.makedirs(os.path.join(data_folder, "train"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "val"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "test"), exist_ok=True)
    
    with open(os.path.join(data_folder, "train", "QA_data.json"), "w") as f:
        json.dump(train_QA_data, f)
    with open(os.path.join(data_folder, "train", "paper_data.json"), "w") as f:
        json.dump(train_paper_data, f)
    with open(os.path.join(data_folder, "val", "QA_data.json"), "w") as f:
        json.dump(val_QA_data, f)
    with open(os.path.join(data_folder, "val", "paper_data.json"), "w") as f:
        json.dump(val_paper_data, f)
    with open(os.path.join(data_folder, "test", "QA_data.json"), "w") as f:
        json.dump(test_QA_data, f)
    with open(os.path.join(data_folder, "test", "paper_data.json"), "w") as f:
        json.dump(test_paper_data, f)
    if len(train_QA_data) == 2593 and len(val_QA_data) == 1005\
         and len(train_paper_data) == 888 and len(val_paper_data) == 281:
        print("correctly preprocessed the QASPER dataset, aligning with the reported data.")
    else:
        print("incorrectly preprocessed the QASPER dataset, not aligning with the reported data.")
    
    print("preprocess QASPER done")

if __name__ == "__main__":
    preprocess_QASPER("./data/qasper")