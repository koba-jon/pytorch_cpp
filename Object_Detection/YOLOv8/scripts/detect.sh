#!/bin/bash

DATA='VOC2012'

./YOLOv8 \
    --detect true \
    --model "yolov8s" \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 20 \
    --size 640 \
    --prob_thresh 0.03 \
    --gpu_id 0 \
    --nc 3
