#!/bin/bash
echo "Using GPU: $3"
export CUDA_VISIBLE_DEVICES=$3
echo "python train_scripts/train_algo.py -n_data $1 -n_trial $2 -pi gail -Qloss adv -n_score 1 -tau 0.3 -explr_const 0.5 -d_lr 1e-4 -g_lr 1e-4"
python train_scripts/train_algo.py -n_data $1 -n_trial $2 -pi gail -Qloss adv -n_score 1 -tau 0.3 -explr_const 0.5 -d_lr 1e-4 -g_lr 1e-4

