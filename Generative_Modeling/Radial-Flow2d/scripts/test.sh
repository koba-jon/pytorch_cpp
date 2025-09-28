#!/bin/bash

DATA='celebA'

./Radial-Flow2d \
    --test true \
    --dataset ${DATA} \
    --size 32 \
    --gpu_id 0 \
    --nc 3
