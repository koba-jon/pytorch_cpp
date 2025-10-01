#!/bin/bash

DATA='celebA'

./Real-NVP2d \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --size 64 \
    --batch_size 16 \
    --gpu_id 0 \
    --nc 3
