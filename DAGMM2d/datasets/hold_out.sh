#!/bin/bash

python3 hold_out.py \
    --input_dir MVTecAD_org \
    --output_dir MVTecAD \
    --train_rate 9 \
    --valid_rate 1
