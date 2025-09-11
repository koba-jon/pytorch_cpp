#!/bin/bash

DATA='celebA'

./MAE \
    --test true \
    --dataset ${DATA} \
    --test_dir "test" \
    --size 224 \
    --gpu_id 0 \
    --nc 3
