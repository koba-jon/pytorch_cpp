#!/bin/bash

DATA='celebA'

./DDPM2d \
    --sample true \
    --timesteps 10000 \
    --dataset ${DATA} \
    --sample_total 100 \
    --size 256 \
    --gpu_id 0 \
    --nc 3
