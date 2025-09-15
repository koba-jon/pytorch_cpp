#!/bin/bash

DATA='celebA'

./VQ-VAE \
    --train2 true \
    --train2_epochs 1 \
    --dataset ${DATA} \
    --size 256 \
    --train2_batch_size 16 \
    --gpu_id 0 \
    --nc 3
