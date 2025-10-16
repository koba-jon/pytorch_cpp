#!/bin/bash

DATA='MNIST'

./PixelSNAIL-Gray \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --size 64 \
    --batch_size 1 \
    --gpu_id 0
