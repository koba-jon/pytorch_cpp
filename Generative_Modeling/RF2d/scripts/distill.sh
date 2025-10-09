#!/bin/bash

DATA='celebA'

./RF2d \
    --distill true \
    --distill_epochs 300 \
    --dataset ${DATA} \
    --size 256 \
    --distill_batch_size 16 \
    --gpu_id 0 \
    --nc 3
