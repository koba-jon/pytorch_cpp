#!/bin/bash

python3 hold_out.py \
    --input_dir VOC2012_org \
    --output_dir VOC2012 \
    --train_rate 9 \
    --valid_rate 1
