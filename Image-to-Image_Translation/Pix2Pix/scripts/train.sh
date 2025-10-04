#!/bin/bash

DATA='facade'

./Pix2Pix \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --size 256 \
    --loss "vanilla" \
    --batch_size 1 \
    --gpu_id 0 \
    --input_nc 3 \
    --output_nc 3
