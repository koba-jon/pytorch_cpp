#!/bin/bash

DATA='celebA'

./VAE \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --size 256 \
    --loss "l1" \
    --batch_size 16 \
    --gpu_id 0 \
    --nc 3
