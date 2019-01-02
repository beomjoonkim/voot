echo "Using GPU: $3"
export CUDA_VISIBLE_DEVICES=$3
echo "python train_scripts/train_algo.py -n_data $1 -n_trial $2 -a ddpg_new -tau 0.001 -explr_const 0.5"
python train_scripts/train_algo.py -n_data $1 -n_trial $2 -a ddpg_new -tau 0.001 -explr_const 0.5
