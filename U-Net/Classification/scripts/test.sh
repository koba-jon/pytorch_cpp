#!/bin/bash

DATA='VOC2012'

./U-Net_Classification \
	--test true \
	--dataset ${DATA} \
	--size 256 \
	--gpu_id 0 \
	--nc 3
