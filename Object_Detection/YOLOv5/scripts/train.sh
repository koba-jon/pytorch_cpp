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
    --size 608 \
    --batch_size 2 \
    --prob_thresh 0.03 \
    --Lambda_noobject 0.1 \
    --lr_init 1e-4 \
    --lr_base 1e-3 \
    --lr_decay1 1e-4 \
    --lr_decay2 1e-5 \
    --gpu_id 0 \
    --nc 3
