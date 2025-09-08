#!/bin/bash

DATA='MNIST'

./EfficientNet \
    --test true \
    --network "B0" \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 10 \
    --gpu_id 0 \
    --nc 1
