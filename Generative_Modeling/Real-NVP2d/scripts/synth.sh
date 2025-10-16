#!/bin/bash

DATA='celebA'

./Real-NVP2d \
    --synth true \
    --dataset ${DATA} \
    --size 64 \
    --gpu_id 0 \
    --nc 3
