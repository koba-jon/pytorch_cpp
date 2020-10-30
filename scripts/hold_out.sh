#!/bin/bash

python3 ../../../scripts/hold_out.py \
    --input_dir celebA_org \
    --output_dir celebA \
    --train_rate 9 \
    --valid_rate 1
