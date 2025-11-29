#!/bin/bash

DATA='MVTecAD'

./PaDiM \
    --AD true \
    --dataset ${DATA} \
    --normal_path "test_result/image_scoreN.txt" \
    --anomaly_path "test_result/image_scoreA.txt" \
    --AD_result_dir "AD_result/Image-AUROC"

./PaDiM \
    --AD true \
    --dataset ${DATA} \
    --normal_path "test_result/pixel_scoreN.txt" \
    --anomaly_path "test_result/pixel_scoreA.txt" \
    --AD_result_dir "AD_result/Pixel-AUROC"
