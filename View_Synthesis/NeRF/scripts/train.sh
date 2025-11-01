#!/bin/bash

DATA='srn_cars'

./NeRF \
    --train true \
    --epochs 10000 \
    --dataset ${DATA} \
    --size 128 \
    --focal_length 131.25 \
    --batch_size 1 \
    --gpu_id 0
