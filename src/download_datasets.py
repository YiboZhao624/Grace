import argparse
import logging
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict

import requests


logger = logging.getLogger("DatasetDownloader")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def download_file(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        logger.info("File already exists, skipping download: %s", dest_path)
        return

    logger.info("Downloading %s -> %s", url, dest_path)
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with open(dest_path, "wb") as fout:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    fout.write(chunk)


def extract_tar_gz(archive_path: Path, dest_dir: Path) -> None:
    logger.info("Extracting %s", archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)


def extract_zip(archive_path: Path, dest_dir: Path) -> None:
    logger.info("Extracting %s", archive_path)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(path=dest_dir)


def download_qasper(root_dir: Path) -> None:
    qasper_dir = root_dir / "qasper"
    qasper_dir.mkdir(parents=True, exist_ok=True)

    urls: Dict[str, str] = {
        "qasper-train-dev-v0.3.tgz": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
        "qasper-test-and-evaluator-v0.3.tgz": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
    }

    for filename, url in urls.items():
        archive_path = qasper_dir / filename
        download_file(url, archive_path)
        extract_tar_gz(archive_path, qasper_dir)
        archive_path.unlink(missing_ok=True)

    # Move json files to the root qasper directory
    train_dev_subdir = qasper_dir / "qasper-train-dev-v0.3"
    if train_dev_subdir.is_dir():
        for json_name in ["qasper-train-v0.3.json", "qasper-dev-v0.3.json"]:
            shutil.move(str(train_dev_subdir / json_name), str(qasper_dir / json_name))
        shutil.rmtree(train_dev_subdir)

    test_subdir = qasper_dir / "qasper-test-and-evaluator-v0.3"
    if test_subdir.is_dir():
        shutil.move(str(test_subdir / "qasper-test-v0.3.json"), str(qasper_dir / "qasper-test-v0.3.json"))
        shutil.rmtree(test_subdir)

    logger.info("QASPER download and extraction completed.")


def download_hotpotqa(root_dir: Path) -> None:
    hotpot_dir = root_dir / "hotpot"
    hotpot_dir.mkdir(parents=True, exist_ok=True)

    urls = {
        "hotpot_train_v1.1.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
        "hotpot_dev_distractor_v1.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "hotpot_dev_fullwiki_v1.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
    }

    for filename, url in urls.items():
        download_file(url, hotpot_dir / filename)

    logger.info("HotpotQA download completed.")


def download_two_wiki(root_dir: Path) -> None:
    two_wiki_dir = root_dir / "2wikimultihop"
    two_wiki_dir.mkdir(parents=True, exist_ok=True)

    zip_url = "https://www.dropbox.com/s/npidmtadreo6df2/data.zip?dl=1"
    archive_path = two_wiki_dir / "data.zip"

    download_file(zip_url, archive_path)
    extract_zip(archive_path, two_wiki_dir)
    archive_path.unlink(missing_ok=True)

    logger.info("2WikiMultiHopQA download and extraction completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract QA datasets.")
    parser.add_argument(
        "--dataset",
        choices=["qasper", "hotpotqa", "2wiki", "all"],
        default="all",
        help="Which dataset to download.",
    )
    parser.add_argument(
        "--data-root",
        default="./data",
        help="Root directory to store downloaded datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.data_root)
    root_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in {"qasper", "all"}:
        download_qasper(root_dir)

    if args.dataset in {"hotpotqa", "all"}:
        download_hotpotqa(root_dir)

    if args.dataset in {"2wiki", "all"}:
        download_two_wiki(root_dir)


if __name__ == "__main__":
    main()
