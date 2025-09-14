#!/bin/bash

DATA='MNIST'

./EfficientNet \
    --train true \
    --network "B0" \
    --epochs 300 \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 10 \
    --batch_size 16 \
    --gpu_id 0 \
    --nc 1
