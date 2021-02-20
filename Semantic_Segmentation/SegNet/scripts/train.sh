#!/bin/bash

DATA='VOC2012'

./SegNet \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --class_num 22 \
    --size 256 \
    --batch_size 16 \
    --gpu_id 0 \
    --nc 3
