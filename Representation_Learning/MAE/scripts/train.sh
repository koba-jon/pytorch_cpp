#!/bin/bash

DATA='celebA'

./MAE \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --size 224 \
    --batch_size 16 \
    --gpu_id 0 \
    --nc 3
