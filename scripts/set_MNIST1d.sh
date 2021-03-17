#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

python3 ${SCRIPT_DIR}/set_MNIST1d.py \
    --output1_dir "MNIST1d_org" \
    --output2_dir "MNIST1d"
