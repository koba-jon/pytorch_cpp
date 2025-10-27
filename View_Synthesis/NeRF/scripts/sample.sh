#!/bin/bash

DATA='srn_cars'

./NeRF \
    --sample true \
    --dataset ${DATA} \
    --sample_total 100 \
    --size 128 \
    --focal_length 131.25 \
    --gpu_id 0
