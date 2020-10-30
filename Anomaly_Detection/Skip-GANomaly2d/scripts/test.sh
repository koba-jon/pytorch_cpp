#!/bin/bash

DATA='MVTecAD'

./Skip-GANomaly2d \
    --test true \
    --dataset ${DATA} \
    --test_dir "test_anomaly" \
    --test_result_dir "test_result_anomaly" \
    --size 256 \
    --gpu_id 0 \
    --nc 3

./Skip-GANomaly2d \
    --test true \
    --dataset ${DATA} \
    --test_dir "test_normal" \
    --test_result_dir "test_result_normal" \
    --size 256 \
    --gpu_id 0 \
    --nc 3
