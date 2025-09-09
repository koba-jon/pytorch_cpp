#!/bin/bash

DATA='celebA'

./DDPM2d \
    --synth true \
    --timesteps 10000 \
    --dataset ${DATA} \
    --size 256 \
    --gpu_id 0 \
    --nc 3
