#!/bin/bash

DATA='facade'

./CycleGAN \
    --test true \
    --dataset ${DATA} \
    --size 256 \
    --gpu_id 0 \
    --A_nc 3 \
    --B_nc 3
