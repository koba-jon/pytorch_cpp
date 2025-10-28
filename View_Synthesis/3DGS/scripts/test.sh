#!/bin/bash

DATA='srn_cars'

./3DGS \
    --test true \
    --dataset ${DATA} \
    --size 128 \
    --focal_length 131.25 \
    --gpu_id 0
