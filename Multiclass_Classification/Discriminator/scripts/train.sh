#!/bin/bash

DATA='MNIST'

./Discriminator \
    --train true \
    --BN true \
    --epochs 300 \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 10 \
    --size 256 \
    --batch_size 16 \
    --gpu_id 0 \
    --nc 1
