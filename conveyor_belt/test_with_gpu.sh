#!/bin/bash
echo "Disabling GPUs"
export CUDA_VISIBLE_DEVICES=''
echo "python test_scripts/test_algo.py -a $1 -n_data $2 -n_trial $3 -epoch $4"
python test_scripts/test_algo.py -a $1 -n_data $2 -n_trial $3 -epoch $4
