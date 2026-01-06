# GRACE: Reinforcement Learning for Grounded Response and Abstention under Contextual Evidence

This repository contains the official implementation of GRACE, a reinforcement-learning-based framework for training large language models to produce grounded responses and to abstain appropriately when contextual evidence is insufficient.

## Project Structure

```
├── data/                # Raw and preprocessed datasets, default empty
├── src/                 # Core implementation
├── scripts/             # Reproduction scripts
├── vllm/                # vLLM server launch scripts
├── requirements.txt     # Requirements for the project
└── README.md            # This file
```

## Usage

We recommend using a clean Conda environment. Our experiments were conducted with Python 3.12, CUDA 12.4, and 4 NVIDIA A800-80GB GPUs.

### Create Conda Environment

```
conda create -n GRACE python=3.12
conda activate GRACE
```

### Install Dependencies

To install the dependencies, please run the following command:

```
pip install -r requirements.txt
```

This command only install the extra packages we need except the verl and its dependencies.

Then, please follow the instruction of [verl](https://verl.readthedocs.io/en/latest/start/install.html) to prepare the dependencies of verl. Finally, clone the verl and install the verl with the following command:

```
git clone https://github.com/volcengine/verl
cd verl
git checkout 4aa02fe1663d8048b9c204345b2abe5197870df3
pip install --no-deps -e .
```

We kindly note that we basically did not change the code in verl, but add several lines in `verl/recipe/dapo/dapo_ray_trainer.py` to monitor the training process following this [issue](https://github.com/volcengine/verl/issues/2279).


### Prepare Data

We use the open-sourced open-domain datasets `HotpotQA`, and open-sourced specific domain datasets `qasper`.

- **Download raw datasets**: `scripts/prepare_datasets.sh` downloads QASPER and HotpotQA into `./data/` by default. Override the target folder by modifying the `DATA_ROOT` variable in the script.
- **Preprocess into unified format**: `scripts/preprocess.sh` runs `src/preprocess.py` so that QASPER and HotpotQA all emit `QA_data.json` and `paper_data.json` (when available) under split-specific subfolders.

Example usage:

```
DATA_ROOT=/absolute/path/to/data ./scripts/prepare_datasets.sh
DATA_ROOT=/absolute/path/to/data ./scripts/preprocess.sh
```

### Data Generation

We provide the `data_generation.py`, which includes a few kinds of data generators. If you want to implement more kinds of datagenerator, just inherit the `BaseDatasetGenerator` class. 

To reproduce our data generation process, please first start the vLLM server with the scripts in `./vllm`, and then run the `bash main.sh` under `./scripts` to generate all the data listed in the responding config files.

### Training

We provide the `train.sh` to train the model. You can change the parameters in the `train.sh` to train the model.

The training process in the current version of the verl framework is not fully stable, and setting a fixed random seed cannot completely eliminate the effect of randomness (see (issue #1683)[https://github.com/volcengine/verl/issues/1683] for details). To maximize reproducibility, we will additionally release the trained LoRA adapter and the corresponding model weights once the paper is accepted.

### Evaluation

For evaluation, we provide the `evaluator.py`, which includes the class Evaluator and provide the following metrics:

1. Rouge-L, BLEU: We implement them with the official package `evaluate` by Huggingface.

2. Exact Match: We directly use the code provided by the [ARENA paper](https://arxiv.org/pdf/2505.13258) to calculate the exact match score.

3. BERT Score: We implement it with the official package `bert_score` by Huggingface using the `bert-base-uncased` model.

4. LLM-as-a-Judge: We directly use the prompt provided by the [ARENA paper](https://arxiv.org/pdf/2505.13258), and we call the DeepSeek API as the judge.

You can deploy the vllm server with the trained model, and then leverage the `inference.py` script to inference and evaluate the model. Please note that the evaluation process requires an active internet connection. It is possible that due to the unstable internet connection, the evaluation process may fail. In this case, you can manually start the evaluation process by running the `bash run_evaluate.sh` under the `./script` folder.