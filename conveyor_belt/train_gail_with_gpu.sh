echo "Using GPU: $5"
export CUDA_VISIBLE_DEVICES=$5
echo "python train_scripts/train_algo.py -n_data $1 -n_trial $2 -a gail -tau $3 -explr_const $4"
python train_scripts/train_algo.py -n_data $1 -n_trial $2 -a gail -tau $3 -explr_const $4
