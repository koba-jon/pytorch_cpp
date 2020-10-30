#!/bin/bash

DATA='MNIST'

./VGGNet \
    --test true \
    --n_layers 16 \
    --BN true \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 10 \
    --size 224 \
    --gpu_id 0 \
    --nc 1
