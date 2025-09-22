#!/bin/bash

DATA='celebA'

./PixelCNN-RGB \
    --test true \
    --dataset ${DATA} \
    --size 64 \
    --gpu_id 0
