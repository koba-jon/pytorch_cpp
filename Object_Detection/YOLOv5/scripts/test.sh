#!/bin/bash

DATA='VOC2012'

./YOLOv5 \
    --test true \
    --model "yolov5s" \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 20 \
    --size 608 \
    --prob_thresh 0.03 \
    --Lambda_noobject 0.1 \
    --gpu_id 0 \
    --nc 3
