import numpy as np
import os
import argparse
import tensorflow as tf
import socket
from problem_environments.conveyor_belt_rl_env import RLConveyorBelt
from problem_environments.minimum_displacement_removal_rl import RLMinimumDisplacementRemoval
from problem_instantiators.minimum_constraint_removal_instantiator import MinimumConstraintRemovalInstantiator
from DDPG import DDPG
from PPO import PPO
import time
import sys

# from TRPO import TRPO


hostname = socket.gethostname()
if hostname == 'dell-XPS-15-9560' or hostname == 'phaedra' or hostname == 'shakey' or hostname == 'lab':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/gtamp_results/test_results/'


def setup_save_dirs(parent_dir, domain, pi_name, n_data, n_trial, d_lr, g_lr, tau, nscore_train, explr_p):
    n_data_dir = parent_dir + '/RL_results/'+ domain + '/n_data_' + str(n_data)
    pi_dir = n_data_dir + '/' + pi_name

    lr_dir = pi_dir + '/dg_lr_' + str(d_lr) + '_' + str(g_lr) + '/'
    tau_dir = lr_dir + '/tau_' + str(tau) + '/'
    explr_p_dir = tau_dir + '/' + '/explr_p_' + str(explr_p) + '/'
    nscore_dir = explr_p_dir + '/' + '/n_score_' + str(nscore_train) + '/'
    trial_dir = nscore_dir + '/n_trial_' + str(n_trial)
    train_results_dir = trial_dir + '/train_results/'

    if not os.path.exists(train_results_dir):
        try:
            os.makedirs(train_results_dir)
        except OSError:
            time.sleep(1)
            if not os.path.exists(train_results_dir):
                os.makedirs(train_results_dir)

    return train_results_dir


def create_policy(alg, train_results_dir, tau, explr_p, v):
    session = tf.Session()

    print train_results_dir
    print train_results_dir
    print '========'

    if 'mdr' in train_results_dir:
        dim_state = 3 * 7
        dim_action = 3
    else:
        dim_state = 3 * 20
        dim_action = 9

    if alg.find('ddpg') != -1:
        assert tau is not None, 'ddpg requires tau'
        policy = DDPG(session,
                      dim_action,
                      dim_state,
                      tau=tau,
                      save_folder=train_results_dir,
                      explr_const=explr_p,
                      visualize=v)
    elif alg == 'ppo':
        policy = PPO(session, dim_action, dim_state, tau=tau, save_folder=train_results_dir, explr_const=explr_p)
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
    explr_p = args.explr_p
    n_score_train = args.n_score
    train_results_dir = setup_save_dirs(ROOTDIR, args.domain, alg, n_data, n_trial,
                                        d_lr, g_lr, tau, n_score_train, explr_p)

    policy = create_policy(alg, train_results_dir, tau, explr_p, v)

    print "Starting train"
    if args.domain == 'convbelt':
        epochs = 3000
        problem = RLConveyorBelt(problem_idx=3, n_actions_per_node=3)  # different "initial" state
    else:
        epochs = 2000
        problem = RLMinimumDisplacementRemoval(problem_idx=0)

    policy.train(problem, args.seed, epochs=epochs, d_lr=1e-3, g_lr=1e-4)
    #policy.saveWeights(additional_name='_' + str(args.seed) + '_')
    create_done_marker(train_results_dir, args.seed)


def create_done_marker(train_results_dir, seed):
    fin = open(train_results_dir + '/%d_done_train.txt' % (seed), 'a')
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
    parser.add_argument('-tau', type=float, default=1e-3)
    parser.add_argument('-d_lr', type=float, default=1e-3)
    parser.add_argument('-g_lr', type=float, default=1e-4)
    parser.add_argument('-n_score', type=int, default=5)
    parser.add_argument('-otherpi', default='uniform')
    parser.add_argument('-explr_p', type=float, default=0.3)
    parser.add_argument('-domain', type=str, default='convbelt')
    parser.add_argument('-seed', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_agent(args)


if __name__ == '__main__':
    main()
