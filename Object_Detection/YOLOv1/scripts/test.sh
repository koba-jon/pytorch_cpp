#!/bin/bash

DATA='VOC2012'

./YOLOv1 \
    --test true \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 20 \
    --size 448 \
    --prob_thresh 0.03 \
    --gpu_id 0 \
    --nc 3
