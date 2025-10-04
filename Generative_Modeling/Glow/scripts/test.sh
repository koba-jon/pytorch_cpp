#!/bin/bash

DATA='celebA'

./Glow \
    --test true \
    --dataset ${DATA} \
    --size 64 \
    --gpu_id 0 \
    --nc 3
