# TReSAGe: Transparent Retrieval and Self-knowledge Augmented Generation.

## Usage

We recommend using a clean Conda environment. Our experiments were conducted with Python 3.12, CUDA 12.4, and 4 NVIDIA A800-80GB GPUs.

### Create Conda Environment

```
conda create -n tresage python=3.12
conda activate tresage
```

### Install Dependencies

To install the dependencies, please run the following command:

```
pip install -r requirements.txt
```

Then, please follow the instruction of [verl](https://verl.readthedocs.io/en/latest/start/install.html) to prepare the dependencies of verl. Finally, run the instruction below to install the verl in this repo:

```
cd verl
pip install --no-deps -e .
```

We kindly note that we basically did not change the code in verl, but add several lines in `verl/recipe/dapo/dapo_ray_trainer.py` to monitor the training process following this [issue](https://github.com/volcengine/verl/issues/2279). You can also use the latest version of verl to run our experiments.


### Download Datasets 

We use the open-sourced open-domain datasets including `2wikimultihop`, `hotpotQA`, `musique`, and open-sourced specific domain datasets `qasper` and `peerqa`. To download the datasets, please refer to the `./data/README.md` for more details.

### Data preprocessing

We provide the `preprocess.py`. Please run `python preprocess.py` to preprocess the datasets.

### Data Generation

We provide the `data_generation.py`, which includes a few kinds of data generators. If you want to implement more kinds of datagenerator, just inherit the `QASPERDataGenerator` class and implement the `organize_evidence` method. To reproduce our data generation process, please run the `main.py`. It will generate all the data used in experiments.

### Training

We provide the `train.sh` to train the model. You can change the parameters in the `train.sh` to train the model.

The training process in the current version of the verl framework is not fully stable, and setting a fixed random seed cannot completely eliminate the effect of randomness (see (issue #1683)[https://github.com/volcengine/verl/issues/1683] for details). To maximize reproducibility, we will additionally release the trained LoRA adapter and the corresponding model weights once the paper is accepted.

### Evaluation

For evaluation, we provide the `evaluator.py`, which includes the class Evaluator and provide the following metrics:

1. Rouge-L, BLEU: We implement them with the official package `evaluate` by Huggingface.

2. Exact Match: We directly use the code provided by the [ARENA paper](https://arxiv.org/pdf/2505.13258) to calculate the exact match score.

3. BERT Score: We implement it with the official package `bert_score` by Huggingface using the `bert-base-uncased` model.

4. LLM-as-a-Judge: We directly use the prompt provided by the [ARENA paper](https://arxiv.org/pdf/2505.13258), and we use the openai package to call the GPT-4o-mini API as the judge.

You can deploy the vllm server with the trained LoRA adapter, and then leverage the `inference.py` script to inference and evaluate the model. Please note that the evaluation process requires an active internet connection.

## Contribution Handing

If you are interested in this project and want to contribute, there will be a few aspects you can contribute to:

1. Documentation: We are still working on the documentation of the project. If you are interested in this project and want to contribute, you can help us improve the documentation.

2. Test Code: You can provide the test code for the project.

3. New Feature: If you want to add new features to the project, just implement the new feature and add the corresponding test code to faciliate our code review.

4. Bug Fix: If you find any bugs in the project, please report them to us.

If you want to contribute, feel free to start a pull request. We will reply and review your pull request as soon as possible.