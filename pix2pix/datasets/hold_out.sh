#!/bin/bash

python3 hold_out.py \
    --input_dir facade_org \
    --output_dir facade \
    --train_rate 9 \
    --valid_rate 1
