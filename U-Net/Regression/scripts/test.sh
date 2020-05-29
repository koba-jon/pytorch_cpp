#!/bin/bash

DATA='maps'

./U-Net_Regression \
    --test true \
    --dataset ${DATA} \
    --size 256 \
    --gpu_id 0 \
    --input_nc 3 \
    --output_nc 3
