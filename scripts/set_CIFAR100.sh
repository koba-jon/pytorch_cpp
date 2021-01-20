#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

python3 ${SCRIPT_DIR}/set_CIFAR100.py \
    --output1_dir "CIFAR100_org" \
    --output2_dir "CIFAR100"
