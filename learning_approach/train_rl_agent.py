import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
import socket

from DDPG import DDPG

# from TRPO import TRPO


hostname = socket.gethostname()
if hostname == 'dell-XPS-15-9560' or hostname == 'phaedra' or hostname == 'shakey' or hostname == 'lab':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/gtamp_results/test_results/'


def setup_save_dirs(parent_dir, domain, pi_name, n_data, n_trial, d_lr, g_lr, tau, nscore_train, explr_const):
    n_data_dir = parent_dir + '/RL_results/'+ domain + '/n_data_' + str(n_data)
    pi_dir = n_data_dir + '/' + pi_name

    lr_dir = pi_dir + '/dg_lr_' + str(d_lr) + '_' + str(g_lr) + '/'
    tau_dir = lr_dir + '/tau_' + str(tau) + '/'
    explr_const_dir = tau_dir + '/' + '/explr_const_' + str(explr_const) + '/'
    nscore_dir = explr_const_dir + '/' + '/n_score_' + str(nscore_train) + '/'
    trial_dir = nscore_dir + '/n_trial_' + str(n_trial)
    train_results_dir = trial_dir + '/train_results/'

    if not os.path.exists(train_results_dir):
        os.makedirs(train_results_dir)
    return train_results_dir


def create_policy(alg, train_results_dir, tau, explr_const, v):
    session = tf.Session()

    print train_results_dir
    print train_results_dir
    print '========'

    # todo use the poses of objects first
    dim_state = 3 * 20
    dim_action = 9

    if alg.find('ddpg') != -1:
        assert tau is not None, 'ddpg requires tau'
        policy = DDPG(session,
                      dim_action,
                      dim_state,
                      tau=tau,
                      save_folder=train_results_dir,
                      explr_const=explr_const,
                      visualize=v)
    elif alg == 'trpo':
        policy = TRPO(session,
                      dim_action,
                      dim_state,
                      tau=tau,
                      save_folder=train_results_dir,
                      explr_const=explr_const,
                      visualize=v)
    else:
        raise NotImplementedError

    return policy


def determine_trial(parent_dir):
    trial_numbers = [int(ftrial.split('_')[-1]) for ftrial in os.listdir(parent_dir)]
    if len(trial_numbers) == 0:
        return 0
    return np.max(trial_numbers) + 1


def train_agent(args):
    alg = args.a
    n_data = args.n_data
    n_trial = args.n_trial
    d_lr = args.d_lr
    g_lr = args.g_lr
    tau = args.tau  # epsilon in TRPO, tau in DDPG, lambda in SOAP
    v = args.v
    explr_const = args.explr_const
    n_score_train = args.n_score
    train_results_dir = setup_save_dirs(ROOTDIR, args.domain, alg, n_data, n_trial,
                                        d_lr, g_lr, tau, n_score_train, explr_const)

    policy = create_policy(alg, train_results_dir, tau, explr_const, v)

    print "Starting train"
    if args.domain == 'convbelt':
        epochs = 3000
    else:
        epochs = 2000

    policy.train(args.seed, epochs=epochs, d_lr=1e-3, g_lr=1e-4)

    policy.saveWeights(additional_name='_1_')
    create_done_marker(train_results_dir)


def create_done_marker(train_results_dir):
    fin = open(train_results_dir + '/done_train.txt', 'a')
    fin.write('dummy file to mark done\n')
    fin.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Process configurations')
    parser.add_argument('-n_data', type=int, default=100)
    parser.add_argument('-a', default='ddpg')
    parser.add_argument('-g', action='store_true')
    parser.add_argument('-n_trial', type=int, default=-1)
    parser.add_argument('-i', type=int, default=0)
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-tau', type=float, default=1e-2)
    parser.add_argument('-d_lr', type=float, default=1e-3)
    parser.add_argument('-g_lr', type=float, default=1e-4)
    parser.add_argument('-n_score', type=int, default=5)
    parser.add_argument('-otherpi', default='uniform')
    parser.add_argument('-explr_const', type=float, default=0.0)
    parser.add_argument('-domain', type=str, default='convbelt')
    parser.add_argument('-seed', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_agent(args)


if __name__ == '__main__':
    main()
