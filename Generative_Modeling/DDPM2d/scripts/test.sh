#!/bin/bash

DATA='celebA'

./DDPM2d \
    --test true \
    --timesteps 10000 \
    --dataset ${DATA} \
    --test_in_dir "test" \
    --test_out_dir "test" \
    --size 256 \
    --gpu_id 0 \
    --nc 3
