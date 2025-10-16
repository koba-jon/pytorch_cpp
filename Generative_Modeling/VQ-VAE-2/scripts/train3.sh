#!/bin/bash

DATA='celebA'

./VQ-VAE-2 \
    --train3 true \
    --train3_epochs 300 \
    --dataset ${DATA} \
    --size 256 \
    --train3_batch_size 1 \
    --gpu_id 0 \
    --nc 3
