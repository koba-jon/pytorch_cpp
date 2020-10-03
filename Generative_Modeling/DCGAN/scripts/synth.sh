#!/bin/bash

DATA='celebA'

./DCGAN \
    --synth true \
    --dataset ${DATA} \
    --size 256 \
    --gpu_id 0 \
    --nc 3
