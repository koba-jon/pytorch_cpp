#!/bin/bash

DATA='MNIST'

./VGGNet \
    --train true \
    --n_layers 16 \
    --BN true \
    --epochs 300 \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 10 \
    --size 224 \
    --batch_size 16 \
    --gpu_id 0 \
    --nc 1
