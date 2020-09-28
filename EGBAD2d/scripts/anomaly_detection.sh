#!/bin/bash

DATA='MVTecAD'

./EGBAD2d \
    --AD true \
    --dataset ${DATA} \
    --anomaly_path "test_result_anomaly/anomaly_score.txt" \
    --normal_path "test_result_normal/anomaly_score.txt" \
    --n_thresh 256
