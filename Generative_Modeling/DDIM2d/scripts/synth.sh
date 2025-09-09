#!/bin/bash

DATA='celebA'

./DDIM2d \
    --synth true \
    --timesteps 10000 \
    --timesteps_inf 100 \
    --dataset ${DATA} \
    --size 256 \
    --gpu_id 0 \
    --nc 3
