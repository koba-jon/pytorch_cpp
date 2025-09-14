#!/bin/bash

DATA='facade'

./CycleGAN \
    --train true \
    --epochs 300 \
    --iters 1000 \
    --dataset ${DATA} \
    --size 256 \
    --loss "vanilla" \
    --batch_size 1 \
    --gpu_id 0 \
    --A_nc 3 \
    --B_nc 3
