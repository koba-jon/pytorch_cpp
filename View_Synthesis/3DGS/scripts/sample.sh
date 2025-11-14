#!/bin/bash

DATA='srn_cars'

./3DGS \
    --sample true \
    --dataset ${DATA} \
    --sample_total 100 \
    --size 128 \
    --focal_length 131.25 \
    --sample_radius 1.3 \
    --gpu_id 0
