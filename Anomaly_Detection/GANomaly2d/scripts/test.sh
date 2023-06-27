#!/bin/bash

DATA='MVTecAD'
HEATMAP_MAX=0.1

./GANomaly2d \
    --test true \
    --dataset ${DATA} \
    --test_dir "test_anomaly" \
    --test_result_dir "test_result_anomaly" \
    --heatmap_max ${HEATMAP_MAX} \
    --size 256 \
    --gpu_id 0 \
    --nc 3

./GANomaly2d \
    --test true \
    --dataset ${DATA} \
    --test_dir "test_normal" \
    --test_result_dir "test_result_normal" \
    --heatmap_max ${HEATMAP_MAX} \
    --size 256 \
    --gpu_id 0 \
    --nc 3
