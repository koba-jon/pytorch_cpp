#!/bin/bash

DATA='facade'

./pix2pix \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --size 256 \
    --loss "vanilla" \
    --batch_size 16 \
    --gpu_id 0 \
    --input_nc 3 \
    --output_nc 3
