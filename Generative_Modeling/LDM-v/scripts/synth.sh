#!/bin/bash

DATA='celebA'

./LDM-v \
    --synth true \
    --dataset ${DATA} \
    --size 256 \
    --gpu_id 0 \
    --nc 3
