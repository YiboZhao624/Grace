# README

This is our implementation of `Trust-Align` based on `LLaMA-Factory`. As we did not able to build the environment based on the provided `requirements.txt` in the `Trust-Align` repository, we only used their datasets and hyperparameters.

## Environment

The environment is based on the `LLaMA-Factory` environment. You only need to mannually install the `deepspeed`. The whole command is:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
pip install deepspeed==0.16.9
```

## Data

The datasets are provided by the `Trust-Align` repository. You can download them from the [Trust-Align](https://huggingface.co/datasets/declare-lab/Trust-Data/tree/main/) huggingface repository. You can also get the datasets with the following commands:

```bash
cd LLaMA-Factory/data
wget https://huggingface.co/datasets/declare-lab/Trust-Data/blob/main/Trust-Align/dpo/train.json
mv train.json trust_align.json
```

The dataset will be saved in the `data` directory under the LLaMA-Factory repository. Then, you need to modify the `dataset_info.json` file to add the dataset.

```json
"trust_align":{
    "file_name": "trust_align.json",
    "ranking": true,
    "columns":{
      "prompt": "prompt",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  },
```

Then, you can use the dataset in the `LLaMA-Factory` repository.

## Training

In our experiments, we leverage 2 A800 80Gb GPUs. We have already provided the training configuration files, including `qwen3-4b.yaml` and `deepspeed_zero3.json`. You can need to modify the model path and the output directory. If necessary, you need to modify the path of the configuration files and hyperparameters. Then, you can train the model with the following command:

```bash
export CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_full/qwen3-4b.yaml
```