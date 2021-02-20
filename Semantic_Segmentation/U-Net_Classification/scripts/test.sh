#!/bin/bash

DATA='VOC2012'

./U-Net_Classification \
    --test true \
    --dataset ${DATA} \
    --class_num 22 \
    --size 256 \
    --gpu_id 0 \
    --nc 3
