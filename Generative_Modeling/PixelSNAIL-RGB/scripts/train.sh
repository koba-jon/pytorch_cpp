#!/bin/bash

DATA='celebA'

./PixelSNAIL-RGB \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --size 64 \
    --batch_size 1 \
    --gpu_id 0
