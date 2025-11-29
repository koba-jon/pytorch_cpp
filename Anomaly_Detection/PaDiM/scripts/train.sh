#!/bin/bash

DATA='MVTecAD'

./PaDiM \
    --train true \
    --dataset ${DATA} \
    --size 224 \
    --resnet_path "wide_resnet50_2.pth" \
    --n_layers "w50" \
    --select_dim 550 \
    --gpu_id 0
