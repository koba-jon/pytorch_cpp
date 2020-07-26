#!/bin/bash

python3 k_fold_2pair.py \
    --input_dir1 dataset_dir_pair/input \
    --input_dir2 dataset_dir_pair/output \
    --output_dir dataset_dir_pair_k_fold \
    --output_label1 I \
    --output_label2 O \
    --k 5
