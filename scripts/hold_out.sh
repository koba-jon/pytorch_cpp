#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

python3 ${SCRIPT_DIR}/hold_out.py \
    --input_dir "celebA_org" \
    --output_dir "celebA" \
    --train_rate 9 \
    --valid_rate 1
