from keras.optimizers import *
from keras import initializers
from data_load_utils import format_RL_data

import numpy as np
import os


class PolicySearch:
    def __init__(self, sess, dim_action, dim_state, save_folder, tau, explr_const):
        self.opt_G = Adam(lr=1e-4, beta_1=0.5)
        self.opt_D = Adam(lr=1e-3, beta_1=0.5)
        self.initializer = initializers.glorot_normal()
        self.sess = sess
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.tau = tau
        self.n_weight_updates = 0
        self.save_folder = save_folder
        self.explr_const = explr_const

        self.policy = None
        self.qfcn = None
        self.pfilename = None

    def create_qfcn_and_policy(self):
        raise NotImplementedError

    def create_qfcn(self):
        raise NotImplementedError

    def create_policy(self):
        raise NotImplementedError

    def train(self, problem, seed, epochs=500, d_lr=1e-3, g_lr=1e-4):
        raise NotImplementedError

    def augment_dataset(self, traj_list, states, actions, rewards, sprimes):
        new_s, new_a, new_r, new_sprime, new_sumR, _, new_traj_lengths = format_RL_data(traj_list)
        new_a = new_a
        new_data_obtained = len(new_s) > 0

        if new_data_obtained:
            if states is not None:
                n_new = len(new_s)
                n_dim_state = states.shape[1]
                states = np.r_[states, new_s.reshape((n_new, n_dim_state))]
                actions = np.r_[actions, new_a]
                rewards = np.r_[rewards, new_r]
                sprimes = np.r_[sprimes, new_sprime.reshape((n_new, n_dim_state))]
            else:
                states = new_s
                actions = new_a
                rewards = new_r
                sprimes = new_sprime
        else:
            pass

        if states is not None:
            terminal_state_idxs = np.where(np.sum(np.sum(sprimes, axis=-1), axis=-1) == 0)[0]
            nonterminal_mask = np.ones((sprimes.shape[0], 1))
            nonterminal_mask[terminal_state_idxs, :] = 0
        else:
            nonterminal_mask = None

        return states, actions, rewards, sprimes, nonterminal_mask, new_data_obtained

    def log_traj_performance(self, traj_list, n_remain, epoch, n_data):
        if traj_list == -2:
            avg_J = traj_list
        else:
            try:
                avg_J = np.mean([np.sum(traj['r']) for traj in traj_list])
            except:
                import pdb;pdb.set_trace()

        pfile = open(self.pfilename, 'a')
        pfile.write(str(epoch) + ',' + str(avg_J) + ',' + str(n_remain) + ',' + str(n_data) + '\n')
        pfile.close()
        print 'Score of this policy', avg_J
        return avg_J

    def saveWeights(self, init=True, additional_name=''):
        self.policy.save_weights(self.save_folder + '/policy' + additional_name + '.h5')
        self.qfcn.save_weights(self.save_folder + '/qfcn' + additional_name + '.h5')

    def load_offline_weights(self, weight_f):
        self.policy.load_weights(self.save_folder + weight_f)

    def load_weights(self):
        best_rwd = -np.inf
        for weightf in os.listdir(self.save_folder):
            if weightf.find('policy') == -1: continue
            try:
                rwd = float(weightf.split('_')[-1][0:-3])
            except ValueError:
                continue
            if rwd > best_rwd:
                best_rwd = rwd
                best_weight = weightf
        print "Using initial weight ", best_weight
        self.policy.load_weights(self.save_folder + '/' + best_weight)

    def reset_weights(self, init=True):
        if init:
            self.policy.load_weights('policy_init.h5')
            self.qfcn.load_weights('qfcn_init.h5')
        else:
            self.policy.load_weights(self.save_folder + '/policy.h5')
            self.qfcn.load_weights(self.save_folder + '/qfcn.h5')

