#!/bin/bash

set -e

CONFIG_DIR="./configs/test_config"

if [ ! -d "$CONFIG_DIR" ]; then
  echo "Error: Directory '$CONFIG_DIR' not found."
  exit 1
fi

shopt -s nullglob
for config_file in "$CONFIG_DIR"/*.{yaml,yml}; do
  echo "========================================================================"
  echo " "
  echo "RUNNING EXPERIMENT WITH CONFIG: $config_file"
  echo " "
  echo "========================================================================"
  
  python main.py --config "$config_file"
  
  echo " "
  echo "FINISHED EXPERIMENT WITH CONFIG: $config_file"
  echo " "
done

if ! compgen -G "$CONFIG_DIR"/*.{yaml,yml} > /dev/null; then
    echo "No YAML config files found in '$CONFIG_DIR' directory."
fi

echo "========================================================================"
echo " "
echo "All Data Generated."
echo " "
echo "========================================================================"