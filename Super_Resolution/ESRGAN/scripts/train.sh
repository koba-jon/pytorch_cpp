#!/bin/bash

DATA='celebA'

./ESRGAN \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --hr_size 256 \
    --batch_size 16 \
    --gpu_id 0 \
    --nc 3
