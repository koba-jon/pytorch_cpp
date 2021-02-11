#!/bin/bash

DATA='VOC2012'

./YOLOv1 \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 20 \
    --size 448 \
    --batch_size 16 \
    --prob_thresh 0.03 \
    --lr_init 1e-6 \
    --lr_base 1e-5 \
    --lr_decay1 1e-6 \
    --lr_decay2 1e-7 \
    --gpu_id 0 \
    --nc 3
