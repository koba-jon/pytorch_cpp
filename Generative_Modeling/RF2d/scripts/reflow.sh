#!/bin/bash

DATA='celebA'

./RF2d \
    --reflow true \
    --dataset ${DATA} \
    --reflow_total 10000 \
    --size 256 \
    --gpu_id 0 \
    --nc 3
