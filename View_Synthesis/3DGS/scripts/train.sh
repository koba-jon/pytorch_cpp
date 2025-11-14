#!/bin/bash

DATA='srn_cars'

./3DGS \
    --train true \
    --epochs 1000 \
    --dataset ${DATA} \
    --size 128 \
    --focal_length 131.25 \
    --batch_size 1 \
    --gpu_id 0
