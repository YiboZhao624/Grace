#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"

echo "[prepare_datasets] Using data root: $DATA_ROOT"

"$PYTHON_BIN" "$ROOT_DIR/src/download_datasets.py" --dataset all --data-root "$DATA_ROOT"

echo "[prepare_datasets] Downloads complete. Raw files are ready for preprocessing via src/preprocess.py."
