#!/bin/bash

DATA='celebA'

./DDIM2d \
    --sample true \
    --timesteps 10000 \
    --timesteps_inf 100 \
    --dataset ${DATA} \
    --sample_total 100 \
    --size 256 \
    --gpu_id 0 \
    --nc 3
