#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"

echo "[preprocess] Using data root: $DATA_ROOT"

"$PYTHON_BIN" "$ROOT_DIR/src/preprocess.py" --dataset all --data-root "$DATA_ROOT" --hotpotqa-train-sample-size 2000 --two-wiki-train-sample-size 2000 --hotpotqa-test-sample-size 500 --two-wiki-test-sample-size 500 --train-sample-seed 123456

echo "[preprocess] Preprocessing complete. Normalized files are available under dataset-specific split directories."

