#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

python3 ${SCRIPT_DIR}/xml2txt.py \
    --input_dir "datasets/VOC2012/trainX" \
    --output_dir "datasets/VOC2012/trainO" \
    --class_list "list/VOC2012.txt"
