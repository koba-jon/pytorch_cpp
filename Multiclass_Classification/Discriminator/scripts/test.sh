#!/bin/bash

DATA='MNIST'

./Discriminator \
    --test true \
    --BN true \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 10 \
    --size 256 \
    --gpu_id 0 \
    --nc 1
