#!/bin/bash

DATA='celebA'

./VAE2d \
    --synth true \
    --dataset ${DATA} \
    --size 256 \
    --gpu_id 0 \
    --nc 3
