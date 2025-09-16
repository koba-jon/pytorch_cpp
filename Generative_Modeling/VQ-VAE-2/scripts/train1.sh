#!/bin/bash

DATA='celebA'

./VQ-VAE-2 \
    --train1 true \
    --train1_epochs 300 \
    --dataset ${DATA} \
    --size 256 \
    --train1_batch_size 16 \
    --gpu_id 0 \
    --nc 3
