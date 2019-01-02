echo "Using GPU: $7"
export CUDA_VISIBLE_DEVICES=$7
echo "python train_scripts/train_algo.py -n_data $1 -n_trial $2 -a $3 -d_lr $4 -g_lr $5 -tau $5 -explr_const $6"
python train_scripts/train_algo.py -n_data $1 -n_trial $2 -a $3 -d_lr $4 -g_lr $5 -tau $5 -explr_const $6
