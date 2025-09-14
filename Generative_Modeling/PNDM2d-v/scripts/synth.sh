#!/bin/bash

DATA='celebA'

./PNDM2d-v \
    --synth true \
    --dataset ${DATA} \
    --size 256 \
    --gpu_id 0 \
    --nc 3
