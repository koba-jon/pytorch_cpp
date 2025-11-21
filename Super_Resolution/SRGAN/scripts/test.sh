#!/bin/bash

DATA='celebA'

./SRGAN \
    --test true \
    --dataset ${DATA} \
    --hr_size 128 \
    --gpu_id 0 \
    --nc 3
