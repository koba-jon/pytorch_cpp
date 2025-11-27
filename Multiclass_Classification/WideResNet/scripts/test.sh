#!/bin/bash

DATA='MNIST'

./WideResNet \
    --test true \
    --n_layers 50 \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 10 \
    --size 224 \
    --gpu_id 0 \
    --nc 1
