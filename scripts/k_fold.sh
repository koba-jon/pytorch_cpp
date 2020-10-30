#!/bin/bash

python3 ../../../scripts/k_fold.py \
    --input_dir dataset_dir \
    --output_dir dataset_dir_k_fold \
    --k 5
