#!/bin/bash

DATA='MNIST'

./PixelCNN-Gray \
    --test true \
    --dataset ${DATA} \
    --size 64 \
    --gpu_id 0
