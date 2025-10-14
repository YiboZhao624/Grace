from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import random
from typing import Iterable, List

import pandas as pd


def deduplicate_results(results: Iterable[Iterable[str]]) -> List[int]:
    seen = set()
    unique_results_idx: List[int] = []

    for i, result in enumerate(results):
        key = tuple(result)
        if key not in seen:
            seen.add(key)
            unique_results_idx.append(i)

    return unique_results_idx


def extract_content(entry: dict) -> str:
    try:
        return entry["prompt"][1]["content"]
    except (KeyError, IndexError, TypeError):  # pragma: no cover - defensive
        return ""


def extract_retrieved_chunks(entry: dict) -> List[str]:
    content = extract_content(entry)
    if "<ref>" not in content or "</ref>" not in content:
        return [content.strip()] if content else []

    retrieved = content.split("<ref>", 1)[1].split("</ref>", 1)[0].strip()
    chunks = [text.partition(".")[2].strip() for text in retrieved.split("\n")]
    return [chunk for chunk in chunks if chunk]


def process_dataset(
    dataset_paths: List[str],
    deduplicate: bool,
    deduplicate_type: str,
) -> List[dict]:
    records_per_path = [pd.read_parquet(path).to_dict(orient="records") for path in dataset_paths]

    if not deduplicate:
        combined: List[dict] = []
        for records in records_per_path:
            combined.extend(records)
        return combined

    lengths = {len(records) for records in records_per_path}
    if len(lengths) != 1:
        raise ValueError(
            f"Files within dataset do not share identical lengths: {dataset_paths} -> {sorted(lengths)}"
        )

    aligned = [list(items) for items in zip(*records_per_path)]

    if deduplicate_type == "hard":
        filtered = []
        filtered_cnt = 0
        left_cnt = 0
        for row in aligned:
            seen_contents = set()
            for entry in row:
                content = extract_content(entry)
                if content in seen_contents:
                    filtered_cnt += 1
                    continue
                left_cnt += 1
                seen_contents.add(content)
                filtered.append(entry)
        print(f"[hard] dataset={dataset_paths[0]} filtered={filtered_cnt}, left={left_cnt}")
        return filtered

    filtered = []
    original_length = len(aligned) * len(aligned[0] if aligned else [])
    for row in aligned:
        retrieved_lists = [extract_retrieved_chunks(entry) for entry in row]
        unique_idx = deduplicate_results(retrieved_lists)
        filtered.extend(row[i] for i in unique_idx)
    print(f"[soft] dataset={dataset_paths[0]} before={original_length} after={len(filtered)}")
    return filtered


def infer_dataset_name(path: str) -> str:
    stem = Path(path).stem
    if "-" in stem:
        return stem.split("-", 1)[0]
    return stem


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--path", nargs="+", required=True, help="Input parquet file paths")
    parser.add_argument("--output_path", type=str, required=True, help="Output parquet path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deduplicate",
        dest="deduplicate",
        action="store_true",
        default=True,
        help="Enable deduplication (default)",
    )
    parser.add_argument(
        "--no-deduplicate",
        dest="deduplicate",
        action="store_false",
        help="Disable deduplication",
    )
    parser.add_argument(
        "--deduplicate_type",
        type=str,
        default="hard",
        choices=["hard", "soft"],
    )

    args = parser.parse_args()

    grouped_paths = defaultdict(list)
    for path in args.path:
        dataset = infer_dataset_name(path)
        grouped_paths[dataset].append(path)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    merged_data: List[dict] = []

    for dataset, paths in sorted(grouped_paths.items()):
        print(f"Processing dataset '{dataset}' with {len(paths)} files")
        dataset_records = process_dataset(paths, args.deduplicate, args.deduplicate_type)
        shuffled_records = list(dataset_records)
        rng.shuffle(shuffled_records)

        dataset_output_path = output_path.with_name(
            f"{output_path.stem}-{dataset}{output_path.suffix}"
        )
        pd.DataFrame(shuffled_records).to_parquet(dataset_output_path)
        print(f"Saved dataset '{dataset}' to {dataset_output_path}")

        merged_data.extend(shuffled_records)

    rng.shuffle(merged_data)
    pd.DataFrame(merged_data).to_parquet(output_path)
    print(f"Saved merged dataset to {output_path}")


if __name__ == "__main__":
    main()

