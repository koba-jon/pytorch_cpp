#!/bin/bash

DATA='celebA'

./SimCLR \
    --test true \
    --dataset ${DATA} \
    --test_dir "test" \
    --size 224 \
    --gpu_id 0
