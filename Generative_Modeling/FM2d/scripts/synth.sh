#!/bin/bash

DATA='celebA'

./FM2d \
    --synth true \
    --dataset ${DATA} \
    --size 256 \
    --gpu_id 0 \
    --nc 3
