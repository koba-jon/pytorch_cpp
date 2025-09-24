#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

python3 ${SCRIPT_DIR}/k_fold.py \
    --input_dir "dataset_dir" \
    --output_dir "dataset_dir_k_fold" \
    --k 5
