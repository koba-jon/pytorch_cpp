#!/bin/bash

DATA='celebA'

./Planar-Flow2d \
    --test true \
    --dataset ${DATA} \
    --test_dir "test" \
    --size 32 \
    --gpu_id 0 \
    --nc 3
