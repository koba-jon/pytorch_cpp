#!/bin/bash

DATA='MVTecAD'

./AnoGAN2d \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --size 256 \
    --batch_size 32 \
    --gpu_id 0 \
    --nc 3
