#!/bin/bash

DATA='celebA'

./VQ-VAE \
    --test2 true \
    --dataset ${DATA} \
    --size 256 \
    --gpu_id 0 \
    --nc 3
