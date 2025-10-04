#!/bin/bash

DATA='facade'

./Pix2Pix \
    --test true \
    --dataset ${DATA} \
    --size 256 \
    --gpu_id 0 \
    --input_nc 3 \
    --output_nc 3
