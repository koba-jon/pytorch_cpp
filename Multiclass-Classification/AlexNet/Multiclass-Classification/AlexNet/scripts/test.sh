#!/bin/bash

DATA='MNIST'

./AlexNet \
    --test true \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 10 \
    --size 227 \
    --gpu_id 0 \
    --nc 1
