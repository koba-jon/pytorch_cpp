#!/bin/bash

DATA='celebA'

./DDIM2d \
    --train true \
    --timesteps 10000 \
    --timesteps_inf 100 \
    --epochs 300 \
    --dataset ${DATA} \
    --size 256 \
    --batch_size 16 \
    --gpu_id 0 \
    --nc 3
