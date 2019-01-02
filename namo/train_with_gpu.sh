#!/bin/bash
echo "Using GPU: $5"
export CUDA_VISIBLE_DEVICES=$5
echo "python train_scripts/train_algo.py -n_data $1 -n_trial $2 -pi $3 -Qloss $4 -n_score $6 -d_lr $7 -g_lr $8 -tau $9 -explr_const ${10}"
python train_scripts/train_algo.py -n_data $1 -n_trial $2 -pi $3 -Qloss $4 -n_score $6 -d_lr $7 -g_lr $8 -tau $9 -explr_const ${10}
