#!/bin/bash

DATA='MVTecAD'

./PatchCore \
    --test true \
    --dataset ${DATA} \
    --size 224 \
    --resnet_path "wide_resnet50_2.pth" \
    --n_layers "w50" \
    --coreset_rate 0.01 \
    --gpu_id 0

