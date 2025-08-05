# TReSAGe: Transparent Retrieval and Self-knowledge Augmented Generation.

## Usage:

1. Download the dataset. We use the open-sourced open-domain datasets including `2wikimultihop`, `hotpotQA`, `musique`, and open-sourced specific domain datasets `qasper` and `peerqa`. To download the datasets, please refer to the `./data/README.md` for more details.

2. Data preprocessing: We provide the `preprocess.py`, which includes a few kinds of preprocessors. If you want to implement more kinds of preprocessor, just inherit the `Preprocessor` class and implement the `preprocess` method.

3. Data Generation: We provide the `data_generation.py`, which includes a few kinds of data generators. If you want to implement more kinds of datagenerator, just inherit the `QASPERDataGenerator` class and implement the `organize_evidence` method.