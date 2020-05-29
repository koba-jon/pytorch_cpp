#!/bin/bash

DATA='maps'

./U-Net_Regression \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --size 256 \
    --loss "l1" \
    --batch_size 16 \
    --gpu_id 0 \
    --input_nc 3 \
    --output_nc 3
