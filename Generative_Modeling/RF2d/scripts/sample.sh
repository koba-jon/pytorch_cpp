#!/bin/bash

DATA='celebA'

./RF2d \
    --sample true \
    --dataset ${DATA} \
    --sample_total 100 \
    --size 256 \
    --gpu_id 0 \
    --nc 3
