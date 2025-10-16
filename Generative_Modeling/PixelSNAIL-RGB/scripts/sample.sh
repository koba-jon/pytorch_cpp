#!/bin/bash

DATA='celebA'

./PixelSNAIL-RGB \
    --sample true \
    --dataset ${DATA} \
    --sample_total 100 \
    --size 64 \
    --gpu_id 0
