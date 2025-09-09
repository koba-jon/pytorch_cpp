#!/bin/bash

DATA='celebA'

./DDPM2d \
    --train true \
    --timesteps 10000 \
    --epochs 300 \
    --dataset ${DATA} \
    --size 256 \
    --batch_size 16 \
    --gpu_id 0 \
    --nc 3
