#!/bin/bash

DATA='celebA'

./Real-NVP2d \
    --sample true \
    --dataset ${DATA} \
    --sample_total 100 \
    --size 64 \
    --gpu_id 0 \
    --nc 3
