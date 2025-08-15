# TReSAGe: Transparent Retrieval and Self-knowledge Augmented Generation.

## Usage:

1. Download the dataset. We use the open-sourced open-domain datasets including `2wikimultihop`, `hotpotQA`, `musique`, and open-sourced specific domain datasets `qasper` and `peerqa`. To download the datasets, please refer to the `./data/README.md` for more details.

2. Data preprocessing: We provide the `preprocess.py`. Please run `python preprocess.py` to preprocess the datasets.

3. Data Generation: We provide the `data_generation.py`, which includes a few kinds of data generators. If you want to implement more kinds of datagenerator, just inherit the `QASPERDataGenerator` class and implement the `organize_evidence` method. To reproduce our data generation process, please run the `main.py`.

4. 


## Evaluation:

For evaluation, we provide the `evaluator.py`, which includes the class Evaluator and provide the following metrics:

1. Rouge-L, BLEU: We implement them with the official package `evaluate` by Huggingface.

2. Exact Match: We directly use the code provided by the [ARENA paper](https://arxiv.org/pdf/2505.13258) to calculate the exact match score.

3. BERT Score: We implement it with the official package `bert_score` by Huggingface using the `bert-base-uncased` model.

4. LLM-as-a-Judge: We directly use the prompt provided by the [ARENA paper](https://arxiv.org/pdf/2505.13258), and we use the openai package to call the GPT-4o-mini API as the judge.
