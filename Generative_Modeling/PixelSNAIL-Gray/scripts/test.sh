#!/bin/bash

DATA='MNIST'

./PixelSNAIL-Gray \
    --test true \
    --dataset ${DATA} \
    --size 64 \
    --gpu_id 0
