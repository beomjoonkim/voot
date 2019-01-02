echo "Using GPU: $4"
export CUDA_VISIBLE_DEVICES=$4
echo "python train_scripts/train_algo.py -n_data $1 -n_trial $2 -a adq -tau $3 -explr_const 0.5"
python train_scripts/train_algo.py -n_data $1 -n_trial $2 -a adq -tau $3 -explr_const 0.5
