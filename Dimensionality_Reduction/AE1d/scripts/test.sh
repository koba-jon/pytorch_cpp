#!/bin/bash

DATA='NormalDistribution'

./AE1d \
    --test true \
    --dataset ${DATA} \
    --test_in_dir "test" \
    --test_out_dir "test" \
    --nd 300 \
    --nz 1 \
    --gpu_id 0
