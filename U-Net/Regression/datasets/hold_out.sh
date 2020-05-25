#!/bin/bash

python3 hold_out.py \
	--input_dir maps_org \
	--output_dir maps \
	--train_rate 9 \
	--valid_rate 1
