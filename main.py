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
from preprocess import preprocess_QASPER
from argparse import ArgumentParser
from retriever import get_retriever
from chunker import get_chunker
from configs import PreprocessConfig, RetrieverConfig, ChunkerConfig, DataGeneratorConfigs, RerankerConfig
from data_generation import QASPERDataGenerator, QASPERRetrieverDataGenerator, QASPEROracleDataGenerator, QASPERRetrieverRerankerDataGenerator, get_data_generator
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

    # for each method, run generate_data_for_splits.
    for method_config in config['methods']:
        logger.info(f"Method Config: {method_config}")
        single_config_dict = defaults.copy()

        # Use pop() to get the 'name' and remove it from method_config dict
        method_name = method_config.pop('name')
        # merge the config.
        single_config_dict.update(method_config)

        chunker_cfg = ChunkerConfig(**single_config_dict.pop('chunker_config'))
        retriever_cfg = RetrieverConfig(**single_config_dict.pop('retriever_config')) if 'retriever_config' in single_config_dict else None
        reranker_cfg = RerankerConfig(**single_config_dict.pop('reranker_config')) if 'reranker_config' in single_config_dict else None
        
        data_gen_config = DataGeneratorConfig(
            **single_config_dict,
            chunker_config=chunker_cfg,
            retriever_config=retriever_cfg,
            reranker_config=reranker_cfg,
            method=method_name
        )
        logger.info(f"Configs: {data_gen_config}")
        data_generator = get_data_generator(data_gen_config)

        data = data_generator.generate_data_for_splits()

        # saving
        for split_method_key, entries in data.items():
            current_split = split_method_key.split("_")[0]
            
            # generate the filename and path
            output_filename = generate_filename_from_config(data_gen_config, current_split)
            output_path = os.path.join(output_base_dir, output_filename)
            
            save_parquet(entries, output_path)
            logger.info(f"Successfully generated and saved data to: {output_path}")