#!/bin/bash

DATA='VOC2012'

./YOLOv3 \
    --train true \
    --augmentation true \
    --epochs 300 \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 20 \
    --size 608 \
    --batch_size 4 \
    --prob_thresh 0.03 \
    --Lambda_noobject 0.1 \
    --lr_init 1e-5 \
    --lr_base 1e-4 \
    --lr_decay1 1e-5 \
    --lr_decay2 1e-6 \
    --gpu_id 0 \
    --nc 3
