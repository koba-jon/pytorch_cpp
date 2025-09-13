#!/bin/bash

DATA='celebA'

./DDPM2d-v \
    --sample true \
    --dataset ${DATA} \
    --sample_total 100 \
    --size 256 \
    --gpu_id 0 \
    --nc 3
