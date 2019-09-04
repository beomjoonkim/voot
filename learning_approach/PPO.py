import matplotlib as mpl
# mpl.use('Agg') # do this before importing plt to run with no DISPLAY
import matplotlib.pyplot as plt

from keras.layers import *
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.objectives import *
from keras import backend as K
from keras import initializers
from keras.callbacks import ModelCheckpoint


from PolicySearch import PolicySearch
from functools import partial
import time
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import sys

import numpy as np
import scipy.io as sio


import os

from data_load_utils import format_RL_data
import scipy.signal

INFEASIBLE_SCORE = -sys.float_info.max


def ppo_loss(sumA_weight, old_pi_a, tau):
    # This is actually PPO loss function
    def loss(actions, pi_pred):
        # log prob term is -K.sum(K.square(old_pi_a - actions),axis=-1,keepdims=True)
        p_old = K.exp(-K.sum(K.square(old_pi_a - actions), axis=-1, keepdims=True))
        p_new = K.exp(-K.sum(K.square(pi_pred - actions), axis=-1, keepdims=True))
        p_ratio = p_new / (p_old + 1e-5)

        L_cpi = tf.multiply(sumA_weight, p_ratio)
        clipped = tf.clip_by_value(p_ratio, 1 - tau[0, 0], 1 + tau[0, 0])
        L_clipped = tf.multiply(sumA_weight, clipped)
        L = tf.minimum(L_cpi, L_clipped)
        return -L

    return loss


def discount(x, gamma):
    """
    computes discounted sums along 0th dimension of x.
    inputs
    ------
    x: ndarray
    gamma: float
    outputs
    -------
    y: ndarray with same shape as x, satisfying
        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1
    """
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class PPO(PolicySearch):
    def __init__(self, sess, dim_action, dim_state, save_folder, tau, explr_const):
        PolicySearch.__init__(sess, dim_action, dim_state, save_folder, tau, explr_const)

        # define inputs
        self.x_input = Input(shape=(dim_action,), name='x', dtype='float32')  # action
        self.w_input = Input(shape=dim_state, name='w', dtype='float32')  # collision vector
        self.tau_input = Input(shape=(1,), name='tau', dtype='float32')  # collision vector

        self.a_gen, self.disc, self.DG, = self.createGAN()

    def create_qfcn_and_policy(self):
        disc = self.create_qfcn()
        a_gen, a_gen_output = self.create_policy()
        return a_gen, disc, None

    def create_policy(self):
        init_ = self.initializer
        dropout_rate = 0.25
        dense_num = 64
        n_filters = 64

        H = Dense(dense_num, activation='relu')(self.w_input)
        H = Dense(dense_num, activation='relu')(H)
        a_gen_output = Dense(self.dim_action,
                             activation='linear',
                             init=init_,
                             name='a_gen_output')(H)

        # these two are used for training purposes
        sumAweight_input = Input(shape=(1,), name='sumA', dtype='float32')
        old_pi_a_input = Input(shape=(self.dim_action,), name='old_pi_a', dtype='float32')
        a_gen = Model(input=[self.w_input, sumAweight_input, old_pi_a_input, self.tau_input], output=[a_gen_output])
        a_gen.compile(loss=ppo_loss(sumA_weight=sumAweight_input, old_pi_a=old_pi_a_input, tau=self.tau_input),
                      optimizer=self.opt_G)

        return a_gen, a_gen_output

    def create_qfcn(self):
        init_ = self.initializer
        dropout_rate = 0.25
        dense_num = 64

        # K_H = self.k_input
        XK_H = Concatenate(axis=-1)([self.x_input, self.w_input])

        H = Dense(dense_num, activation='relu')(XK_H)
        H = Dense(dense_num, activation='relu')(H)
        disc_output = Dense(1, activation='linear', init=init_)(H)
        disc = Model(input=[self.w_input], output=disc_output, name='disc_output')
        disc.compile(loss='mse', optimizer=self.opt_D)
        return disc

    def predict(self, x, n_samples=1):
        x = x.reshape((len(x), self.n_key_confs, self.dim_state[1]))
        dummy_sumA = np.zeros((n_samples, 1))
        dummy_old_pi_a = np.zeros((n_samples, self.dim_action))
        tau = np.tile(self.tau, (n_samples, 1))

        if n_samples == 1:
            n = n_samples
            d = self.dim_action
            pred = self.a_gen.predict([x, dummy_sumA, dummy_old_pi_a, tau])
            noise = self.explr_const * np.random.randn(n, d)
            return pred + noise
        else:
            n = n_samples
            d = self.dim_action
            pred = self.a_gen.predict([np.tile(x, (n, 1, 1)), dummy_sumA, dummy_old_pi_a, tau])
            noise = self.explr_const * np.random.randn(n, d)
            return pred + noise

    def compute_advantage_values(self, states, actions, sprimes, rewards, traj_lengths):
        Vsprime = np.array([self.disc.predict(s[None, :])[0, 0] \
                                if np.sum(s) != 0 else 0 for s in sprimes])
        n_data = len(Vsprime)
        Vsprime = Vsprime.reshape((n_data, 1))
        Q = rewards + Vsprime
        V = self.disc.predict(states)
        A = Q - V
        sumA = []
        """
        for i in range(len(A)):
          try:
            sumA.append( discount(A[i:i+traj_lengths[i]],1) )
          except IndexError:
            break
    
        #Astd = A.std()
        #normalizedA = (A - A.mean()) / Astd if not np.isclose(Astd,0) else A
        normalizedA = A
        return normalizedA
        """
        for i in range(len(A)):
            try:
                sumA.append(np.sum(A[i:i + traj_lengths[i]]))
            except IndexError:
                break
        return np.array(sumA)[:, None]

    def update_V(self, states, sumR):
        n_data = states.shape[0]
        batch_size = np.min([32, int(len(states) * 0.1)])
        if batch_size == 0:
            batch_size = 1

        checkpointer = ModelCheckpoint(filepath=self.save_folder + '/weights.hdf5',
                                       verbose=0,
                                       save_best_only=True,
                                       save_weights_only=True)
        self.disc.fit(states, sumR, epochs=20,
                      validation_split=0.1,
                      callbacks=[checkpointer],
                      batch_size=batch_size,
                      verbose=False)
        self.disc.load_weights(self.save_folder + '/weights.hdf5')

    def update_policy(self, states, actions, adv):
        n_data = states.shape[0]
        batch_size = np.min([32, int(len(actions) * 0.1)])
        if batch_size == 0:
            batch_size = 1
        tau = np.tile(self.tau, (n_data, 1))
        old_pi_a = self.a_gen.predict([states, adv, actions, tau])
        checkpointer = ModelCheckpoint(filepath=self.save_folder + '/pi_weights.hdf5',
                                       verbose=0,
                                       save_best_only=True,
                                       save_weights_only=True)
        print "Fitting pi..."
        tau = np.tile(self.tau, (n_data, 1))
        self.a_gen.fit([states, adv, old_pi_a, tau],
                       actions, epochs=20, validation_split=0.1,
                       batch_size=batch_size, callbacks=[checkpointer],
                       verbose=False)
        print "Done!"
        self.a_gen.load_weights(self.save_folder + '/pi_weights.hdf5')

    def train(self, problem, seed, epochs=500, d_lr=1e-3, g_lr=1e-4):
        K.set_value(self.opt_G.lr, g_lr)
        K.set_value(self.opt_D.lr, d_lr)

        print self.opt_G.get_config()

        pfilename = self.save_folder + '/' + str(seed) + '_performance.txt'
        pfile = open(pfilename, 'wb')

        for i in range(1, epochs):
            self.epoch = i
            print "N simulations %d/%d" % (i, epochs)
            traj_list = []
            if 'convbelt' in problem.name:
                length_of_rollout = 20
            else:
                length_of_rollout = 10
            for n_iter in range(1):  # N = 5, T = 20, using the notation from PPO paper
                problem.init_saver.Restore()
                problem.objects_currently_not_in_goal = problem.objects
                traj = problem.rollout_the_policy(self, length_of_rollout)
                traj_list.append(traj)
            self.log_traj_performance(traj_list, i)

            # Add new data to the buffer
            new_s, new_a, new_r, new_sprime, new_sumR, _, new_traj_lengths = format_RL_data(traj_list)

            self.update_V(new_s, new_sumR)
            new_sumA = self.compute_advantage_values(new_s, new_a, new_sprime, new_r, new_traj_lengths)
            self.update_policy(new_s, new_a, new_sumA)


