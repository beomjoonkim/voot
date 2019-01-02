import matplotlib as mpl
#mpl.use('Agg') # do this before importing plt to run with no DISPLAY
import matplotlib.pyplot as plt

from keras.layers import *
from keras.layers.merge import Concatenate,Multiply
from keras.models import Sequential, Model
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.objectives import *
from keras import backend as K
from keras import initializers
from keras.callbacks import *
from functools import partial
import time
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import sys

import numpy as np
import scipy.io as sio

import pickle
import math
import csv

import random 
import pandas as pd

def dummy_loss(y_true,y_pred):
  # dummy loss for gps loss
  return K.zeros(1,1)

"""
n_trajs_per_iter=10
n_plan_data = 2112
Sstar_flags = np.zeros((n_plan_data+T*n_trajs_per_iter,))
Sstar_flags[0:n_plan_data] = True
Sstar_flags[n_plan_data:] = False
"""

class PlaceGPS():
  def __init__(self,sess,dim_action,dim_state,n_plan_data,\
               n_traj,time_steps,\
               save_folder,
               key_configs=None,x_scaler=None,c_scaler=None):
    self.initializer = initializers.glorot_normal()
    self.sess = sess
    self.key_configs = key_configs
    self.save_folder=save_folder
    self.s_scaler = x_scaler

    self.n_traj = n_traj
    self.time_steps = time_steps
    self.n_plan_data = n_plan_data

    self.noise_term_var = 0.25 #y= x+L * z, Z~N(0,1), then x ~ N(x,L^2)?
    self.setup_data_dimensions(dim_action,dim_state)
    self.setup_inputs()
    self.create_policy()

  def gps_loss(self,traj_reward,w_r):
    # weights = r(tau) * ImportanceWeights(tau)
    # TODO add the regularizer, log Z(\theta)
    #      I would need to keep track of (u_t - mu_theta)/(u_t-mu_theta')
    n_plan_data = self.n_plan_data
    n_traj      = self.n_traj
    Sstar_flags = np.zeros((n_plan_data+self.time_steps*n_traj,))
    Sstar_flags[0:n_plan_data] = True
    Sstar_flags[n_plan_data:]  = True

    def loss( A,Ahat ):
      Ahat_star     = tf.boolean_mask(Ahat,Sstar_flags==True) 
      Astar         = tf.boolean_mask(A,Sstar_flags==True)
      Ahat_regular  = tf.boolean_mask(Ahat,Sstar_flags==False) 
      A_regular     = tf.boolean_mask(A,Sstar_flags==False)
      w_r_star = tf.boolean_mask(w_r,Sstar_flags==True)
      traj_reward_regular = tf.boolean_mask(traj_reward,Sstar_flags==0) 
      n_reg_data  = tf.cast(tf.shape(Ahat_regular)[0],tf.float32)
      n_star_data = tf.cast(tf.shape(Ahat_star)[0],tf.float32)

      # log exp(-(a-mu(s)))
      RL_term = -(tf.reduce_sum(tf.square(Ahat_regular-A_regular),axis=-1)) 
      # -(a-mu(s))*r(s)
      J = tf.reduce_sum(tf.multiply(tf.transpose(traj_reward_regular),RL_term)) \
                / (n_reg_data +n_star_data)
    
      # log exp(-(a-mu(s)))
      demonstration_term = -(tf.reduce_sum(tf.square(Ahat_star-Astar),axis=-1)) 


      # cost RL = reward of traj  * regression term
      reg = tf.reduce_sum(demonstration_term) / n_star_data


      return -(J + w_r[0]*reg)
    return loss

  def setup_data_dimensions(self,dim_action,dim_state):
    self.dim_action  = dim_action
    self.dim_state   = dim_state
    self.n_key_confs = dim_state[0]

  def setup_inputs(self):
    self.a_input = Input( shape=(self.dim_action,),name='a',dtype='float32' )  
    self.s_input = Input( shape = self.dim_state,name='s',dtype='float32')
    self.r_input = Input( shape = (1,), name='r',dtype='float32')   # rewards
    
    self.astar_input = Input( shape=(self.dim_action,),name='astar',dtype='float32' )  
    self.w_r_input = Input( shape = (1,), name='w_r',dtype='float32')   # rewards
    
  def predict(self,x,n_samples=1):
    if n_samples==1:
      n = x.shape[0]
      d = self.dim_action
      return self.policy.predict(x) + self.noise_term_var*np.random.randn(n,d)
    else:
      n = x.shape[0]
      assert n==1,'Generating multiple actions requires one state'  
      d = self.dim_action
      return self.policy.predict( np.tile(x,(n,1))) + self.noise_term_var*np.random.randn(n,d)
  
  def create_policy(self):
    init_ = self.initializer
    dropout_rate = 0.25
    dense_num = 64
    n_filters=64
    H = Conv2D(filters=n_filters,\
               kernel_size=(1,self.dim_state[1]),\
               strides=(1,1),
               activation='relu')(self.s_input)
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H1=H
    H1 = MaxPooling2D(pool_size=(2,1))(H1)
    H = Flatten()(H1)
    H = Dense(dense_num,activation='relu')(H)
    H = Dense(dense_num,activation='relu')(H)
    a_gen_output = Dense(self.dim_action,
                         activation='linear',
                         init=init_,
                         name='a_gen_output')(H) 
    self.policy = Model(input=[self.s_input],\
                        output=a_gen_output,name='policy')
    self.policy.compile(loss='mse',optimizer='adadelta' )

    self.policy_and_weights = Model( input=[self.s_input,\
                                            self.r_input,\
                                            self.w_r_input],
                                     output=[a_gen_output] )
    self.policy_and_weights.compile(loss=[self.gps_loss(traj_reward=self.r_input,\
                                                   w_r=self.w_r_input)],\
                                    optimizer='adam' )

  def format_policy_search_data( self,trajs ):
    # this function creates state action and sum of rewards of each tracj
    # R(\tau),s,a
    sum_r_traj = []
    for tau in trajs:
      r = tau['r']
      for s,a in zip(tau['s'],tau['a']):
        if 'states' in locals():  
          states = np.r_[states,s]
          actions = np.r_[actions,a]
        else:
          states=s
          actions=a
        sum_r_traj.append( np.sum(r) )
    return np.array(states),np.array(actions),np.array(sum_r_traj) 

  
  def guided_train(self,s,a):
    # - Maximizes the likelihood of the guiding samples (line 3 of GPS)
    # - In the Gaussian case, this is equivalent to minimizing the mse
    #s,a = self.format_guiding_samples( guiding_samples )
    stop = EarlyStopping(monitor='val_loss', patience=10)
    checkpt = ModelCheckpoint(filepath=self.save_folder\
                                +'/guided_trained_weights.{epoch:02d}.{val_loss:.2f}.hdf5',\
                                monitor='loss',\
                                verbose=0, save_best_only=False, \
                                save_weights_only=True)
    hist = self.policy.fit(x=s,y=a,\
                           batch_size=32,\
                           validation_split=0.1,\
                           epochs=200,\
                           callbacks=[checkpt,stop])

  def load_guided_weights(self):
    best_loss = np.inf
    for weightf in os.listdir(self.save_folder):
      if weightf.find('guided_trained_weights')==-1: continue
      loss = float(weightf[-9:-5])
      print weightf,loss,best_loss
      if loss < best_loss:
        best_loss   = loss
        best_weight = weightf
    print "Using initial weight ",best_weight
    self.policy.load_weights(self.save_folder+'/'+best_weight)

  def load_weights(self):
    best_rwd = -np.inf
    for weightf in os.listdir(self.save_folder):
      if weightf.find('policy_search')==-1: continue  
      try:
        rwd=float(weightf.split('_')[-1][0:3])
      except ValueError:
        continue
      if rwd > best_rwd:
        best_rwd   = rwd
        best_weight = weightf
    print "Using policy search weight ",best_weight
    self.policy.load_weights(self.save_folder+'/'+best_weight)

  def save_weights(self,fname):
    self.policy.save_weights(self.save_folder+'/'+fname)
    
  def compute_log_gaussian_prob( self,S,A ):
    return np.norm(A - self.policy.predict(S))^2

  def compute_importance_weights( self,S,A,theta_star ): 
    #  we cannot compute a distribution of a trajectory,
    #  since we cannot obtain the probability of a trajectory
    #  when the trajectory is from a planner.

    # TODO Re-use the samples from past policies
    #self.policy.load_weights(theta_star) 
    #pr_sa = compute_log_gaussian_prob(S,A)
    return np.ones(S.shape[0])
    
  def sample_using_importance( self,S,A,n_data ):
    Ahat = self.policy.predict(S)
    P = np.exp(-np.sum(np.square(Ahat-A),axis=-1))/\
          np.sum(np.exp(-np.sum(np.square(Ahat-A),axis=-1)))
    return np.random.choice(len(S),n_data,p=P)
    
      
  def RL_train( self, trajs, Sstar,Astar, episode_number, theta_star, w_r ):
    S,A,R = self.format_policy_search_data( trajs )
    Rstar   = np.ones((Sstar.shape[0]))*5 # successful trajs have reward of 5
  

    idxs=self.sample_using_importance( S,A,n_data=128 )
    
    S = S[idxs,:]
    A = A[idxs,:]
    R = R[idxs,]
    aggregated_S = np.r_[Sstar,S]
    aggregated_A = np.r_[Astar,A]
    aggregated_R = np.r_[Rstar,R]

    #TODO; importance weights and sampling from them.
    w_r = np.ones((aggregated_S.shape[0]))*w_r
    self.policy_and_weights.fit({'s':aggregated_S,\
                                 'r':aggregated_R,\
                                 'w_r':w_r},
                                 aggregated_A,\
                                 batch_size=len(aggregated_S),\
                                 epochs=1)

  def test_loss_fcn( self, trajs, planner_trajs, episode_number, theta_star, w_r ):
    S,A,R = self.format_policy_search_data( trajs )
    Sstar,Astar = self.format_guiding_samples( planner_trajs )
    Rstar       = np.ones((Sstar.shape[0]))*5

    aggregated_S = np.r_[Sstar,S]
    aggregated_A = np.r_[Astar,A]
    aggregated_R = np.r_[Rstar,R]

    w_r = np.ones((aggregated_S.shape[0]))*w_r
    sigma=0.25

    Ahat         = self.policy.predict(aggregated_S)
    Ahat_regular  = Ahat[Sstar_flags==False,:] 
    A_regular     = aggregated_A[Sstar_flags==False,:]
    gauss_term = (-np.sum(np.square(Ahat_regular-A_regular),axis=-1)) * 1./(2*sigma)
  
    Ahat_star = Ahat[Sstar_flags==True,:]
    Astar     = aggregated_A[Sstar_flags==True,:]
    demonstration_term = (-np.sum(np.square(Ahat_star-Astar),axis=-1)) * 1./(2*sigma)

    n_reg_data = float(Ahat_regular.shape[0])
    n_star_data = float(Ahat_star.shape[0])

    w_r_star = w_r[Sstar_flags==True]
    traj_reward_regular = -aggregated_R[Sstar_flags==False]
    L_RL = np.dot(traj_reward_regular,gauss_term) / n_reg_data

    L_planner = np.sum(demonstration_term)/n_star_data
    L = L_RL - w_r[0]*L_planner

    evals = self.policy_and_weights.evaluate(x={'s':aggregated_S,\
                                                'r':aggregated_R,\
                                                'w_r':w_r},
                                            y=aggregated_A,\
                                            batch_size=aggregated_S.shape[0])
    import pdb;pdb.set_trace()
    assert(np.isclose(evals,L))

