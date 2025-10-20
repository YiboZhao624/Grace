#!/bin/bash
set -euo pipefail

ROOT_DIR="/root/projects/RAGRL"
DATA_DIR="${ROOT_DIR}/data/new_code/top3"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "Directory not found: ${DATA_DIR}" >&2
    exit 1
fi

mapfile -d '' candidates < <(find "${DATA_DIR}" -maxdepth 1 -type f -print0)

train_files=()

for file in "${candidates[@]}"; do
    filename="$(basename "${file}")"
    IFS='-' read -r -a parts <<< "${filename}"
    if (( ${#parts[@]} >= 2 )) && [[ "${parts[1]}" == train ]]; then
        train_files+=("${file}")
    fi
done

if (( ${#train_files[@]} == 0 )); then
    echo "No train files found in ${DATA_DIR}" >&2
    exit 1
fi

python "${ROOT_DIR}/src/merge_and_shuffle.py" \
    --path "${train_files[@]}" \
    "$@" \
    --output_path "${ROOT_DIR}/data/merged/1013-3datasets-soft-deduplicated-top3-0.5.parquet" \
    --deduplicate_type soft