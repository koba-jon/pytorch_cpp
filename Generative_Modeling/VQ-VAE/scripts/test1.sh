#!/bin/bash

DATA='celebA'

./VQ-VAE \
    --test1 true \
    --dataset ${DATA} \
    --test1_in_dir "test1" \
    --test1_out_dir "test1" \
    --size 256 \
    --gpu_id 0 \
    --nc 3
