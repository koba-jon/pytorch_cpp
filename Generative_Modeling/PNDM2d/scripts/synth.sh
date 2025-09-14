#!/bin/bash

DATA='celebA'

./PNDM2d \
    --synth true \
    --dataset ${DATA} \
    --size 256 \
    --gpu_id 0 \
    --nc 3
