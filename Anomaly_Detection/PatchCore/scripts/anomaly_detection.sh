#!/bin/bash

DATA='MVTecAD'

./PatchCore \
    --AD true \
    --dataset ${DATA} \
    --normal_path "test_result/image_scoreN.txt" \
    --anomaly_path "test_result/image_scoreA.txt" \
    --AD_result_dir "AD_result/Image-AUROC"

./PatchCore \
    --AD true \
    --dataset ${DATA} \
    --normal_path "test_result/pixel_scoreN.txt" \
    --anomaly_path "test_result/pixel_scoreA.txt" \
    --AD_result_dir "AD_result/Pixel-AUROC"
