#!/bin/bash

DATA='celebA'

./DCGAN \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --size 256 \
    --loss "vanilla" \
    --batch_size 16 \
    --gpu_id 0 \
    --nc 3
