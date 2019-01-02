#!/bin/bash
echo "Disabling GPUs"
export CUDA_VISIBLE_DEVICES=''
echo "python test_scripts/test_algo.py -n_data $1 -n_trial $2 -pi $3 -Qloss $4 -n_score $5 -epoch $6 -d_lr $7 -g_lr $8 -otherpi $9 -explr_const ${10}"
python test_scripts/test_algo.py -n_data $1 -n_trial $2 -pi $3 -Qloss $4 -n_score $5 -epoch $6 -d_lr $7 -g_lr $8 -otherpi $9 -explr_const ${10}
