#!/bin/bash

DATA='celebA'

./Planar-Flow2d \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --size 32 \
    --batch_size 16 \
    --gpu_id 0 \
    --nc 3
