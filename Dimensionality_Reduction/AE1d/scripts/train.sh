#!/bin/bash

DATA='NormalDistribution'

./AE1d \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --nd 300 \
    --nz 1 \
    --batch_size 64 \
    --gpu_id 0
