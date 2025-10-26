#!/bin/bash

DATA='celebA'

./NeRF \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --size 256 \
    --batch_size 16 \
    --gpu_id 0 \
    --nc 3
