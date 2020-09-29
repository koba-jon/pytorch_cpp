#!/bin/bash

DATA='celebA'

./DAE2d \
    --test true \
    --RVIN true \
    --SPN false \
    --GN false \
    --dataset ${DATA} \
    --size 256 \
    --gpu_id 0 \
    --nc 3
