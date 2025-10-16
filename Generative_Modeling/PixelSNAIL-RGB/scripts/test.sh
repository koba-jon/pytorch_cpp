#!/bin/bash

DATA='celebA'

./PixelSNAIL-RGB \
    --test true \
    --dataset ${DATA} \
    --size 64 \
    --gpu_id 0
