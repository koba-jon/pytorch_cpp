#!/bin/bash

DATA='celebA'

./ESRGAN \
    --test true \
    --dataset ${DATA} \
    --hr_size 256 \
    --gpu_id 0 \
    --nc 3
