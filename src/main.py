# this is the main file for the data generation.
# You can use this file for generating the data for the veRL training stage.
# the optional stage is the difficulty test stage.
# the main stages including:
# 0. parse the arguments.
# 1. preprocess the data.
# 2. initialize the chunker, retriever, and then a kind of data generator.
# 3. generate the data.
# 4. save the data.
# 5. loop the process 2 - 4 for generating more kinds of data.
# 6. test the difficulty of each kind of data.

import yaml
import os
import random
from argparse import ArgumentParser
from configs import RetrieverConfig, ChunkerConfig, RerankerConfig
from data_generation import get_data_generator
from utils import save_parquet, setup_logging
from configs import (DataGeneratorConfig, ChunkerConfig, 
                     RetrieverConfig, RerankerConfig)

logger = setup_logging("Data Generation")

def load_config(path="configs/example_configs.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def generate_filename_from_config(config: DataGeneratorConfig, split: str) -> str:
    # 1. dataset and method.
    dataset = config.dataset.upper()
    method = config.method
    
    # 2. organize the chunker setting.
    chunker_cfg = config.chunker_config
    if chunker_cfg.chunking_strategy == 'token':
        chunker_str = f"{chunker_cfg.max_length}cs"
    else:
        chunker_str = chunker_cfg.chunking_strategy
        
    # 3. organize the retriever setting.
    retriever_str = "None"
    if config.retriever_config:
        r_cfg = config.retriever_config
        if r_cfg.retriever_type == 'bm25':
            retriever_str = 'bm25'
        elif r_cfg.retriever_type in ['vllm', 'sentence_transformer']:
            model_name = r_cfg.model_name.split('/')[-1].replace('-Embedding', '')
            retriever_str = f"{r_cfg.retriever_type}_{model_name}"
        else:
            retriever_str = r_cfg.retriever_type

    # 4. organize the reranker setting.
    reranker_str = "None"
    if config.reranker_config:
        rr_cfg = config.reranker_config
        model_name = rr_cfg.model_name.split('/')[-1].replace('BAAI-', '').replace("-v2-m3","")
        reranker_str = f"{rr_cfg.reranker_type}_{model_name}"

    # 5. organize the topk setting.
    top_k_str = f"top{config.top_k}"
    wogt_rate_str = f"wogt{str(config.wo_gt_evidence_rate).replace('.', '')}"

    # 6. organize the filename.
    parts = [
        dataset,
        split,
        method,
        chunker_str,
        retriever_str,
        reranker_str,
        top_k_str,
        wogt_rate_str
    ]

    filename = "-".join(p for p in parts) + ".parquet"
    
    return filename


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", type=str, default="configs/example_configs.yaml")
    args = arg_parser.parse_args()

    config = load_config(args.config)
    defaults = config['data_generator_defaults']
    output_base_dir = config['output_base_dir']

    dataset_entries = config.get('datasets')
    if dataset_entries is None:
        dataset_entries = [
            {
                'dataset': defaults.get('dataset', 'qasper'),
                'data_folder': defaults.get('data_folder', './data/qasper'),
                'split': defaults.get('split', ['train', 'val', 'test'])
            }
        ]

    # for each dataset-method combination, run generate_data_for_splits.
    for dataset_cfg in dataset_entries:
        for method_cfg in config['methods']:
            logger.info(f"Dataset Config: {dataset_cfg}")
            logger.info(f"Method Config: {method_cfg}")

            combined_cfg = defaults.copy()
            combined_cfg.update(dataset_cfg)

            method_name = method_cfg['name']
            retriever_type = method_cfg.get('retriever_config', {'retriever_type': 'Oracle'})['retriever_type']

            # set dataset-specific seed to keep randomness consistent per dataset/method pair
            base_seed = 42
            method_seed = base_seed + hash(f"{dataset_cfg['dataset']}-{method_name}-{retriever_type}") % 1000000
            random.seed(method_seed)
            logger.info(
                "Set unique random seed to %s for dataset '%s' method '%s %s'",
                method_seed,
                dataset_cfg['dataset'],
                method_name,
                retriever_type,
            )

            merged_method_cfg = method_cfg.copy()
            merged_method_cfg.pop('name')
            combined_cfg.update(merged_method_cfg)

            chunker_cfg_dict = combined_cfg.pop('chunker_config')
            chunker_cfg = ChunkerConfig(**chunker_cfg_dict)

            retriever_cfg = None
            if 'retriever_config' in combined_cfg:
                retriever_cfg = RetrieverConfig(**combined_cfg.pop('retriever_config'))

            reranker_cfg = None
            if 'reranker_config' in combined_cfg:
                reranker_cfg = RerankerConfig(**combined_cfg.pop('reranker_config'))

            data_gen_config = DataGeneratorConfig(
                **combined_cfg,
                chunker_config=chunker_cfg,
                retriever_config=retriever_cfg,
                reranker_config=reranker_cfg,
                method=method_name
            )

            logger.info(f"Resolved generator config: {data_gen_config}")
            data_generator = get_data_generator(data_gen_config)

            data = data_generator.generate_data_for_splits()

            for split_method_key, entries in data.items():
                current_split = split_method_key.split("_")[0]
                output_filename = generate_filename_from_config(data_gen_config, current_split)
                output_path = os.path.join(output_base_dir, output_filename)
                save_parquet(entries, output_path)
                logger.info(f"Successfully generated and saved data to: {output_path}")