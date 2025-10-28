#!/bin/bash

DATA='srn_cars'

./3DGS \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --size 128 \
    --focal_length 131.25 \
    --batch_size 1 \
    --gpu_id 0
