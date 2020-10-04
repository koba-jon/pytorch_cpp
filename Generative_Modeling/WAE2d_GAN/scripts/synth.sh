#!/bin/bash

DATA='celebA'

./WAE2d_GAN \
    --synth true \
    --dataset ${DATA} \
    --size 256 \
    --gpu_id 0 \
    --nc 3
