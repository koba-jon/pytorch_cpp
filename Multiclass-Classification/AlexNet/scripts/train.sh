#!/bin/bash

DATA='MNIST'

./AlexNet \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 10 \
    --size 224 \
    --batch_size 64 \
    --gpu_id 0 \
    --nc 1
