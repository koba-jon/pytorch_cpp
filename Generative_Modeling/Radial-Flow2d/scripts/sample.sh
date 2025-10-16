#!/bin/bash

DATA='celebA'

./Radial-Flow2d \
    --sample true \
    --dataset ${DATA} \
    --sample_total 100 \
    --size 32 \
    --gpu_id 0 \
    --nc 3
