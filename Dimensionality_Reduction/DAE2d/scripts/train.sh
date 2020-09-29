#!/bin/bash

DATA='celebA'

./DAE2d \
    --train true \
    --RVIN true \
    --SPN false \
    --GN false \
    --epochs 300 \
    --dataset ${DATA} \
    --size 256 \
    --batch_size 16 \
    --gpu_id 0 \
    --nc 3
