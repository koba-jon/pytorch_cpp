#!/bin/bash

DATA='VOC2012'

./YOLOv5 \
    --train true \
    --augmentation true \
    --model "yolov5s" \
    --epochs 300 \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 20 \
    --size 640 \
    --batch_size 16 \
    --prob_thresh 0.03 \
    --gpu_id 0 \
    --nc 3
