#!/bin/bash

DATA='celebA'

./Glow \
    --synth true \
    --dataset ${DATA} \
    --size 64 \
    --gpu_id 0 \
    --nc 3
