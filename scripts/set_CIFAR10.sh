#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

python3 ${SCRIPT_DIR}/set_CIFAR10.py \
    --output1_dir "CIFAR10_org" \
    --output2_dir "CIFAR10"
