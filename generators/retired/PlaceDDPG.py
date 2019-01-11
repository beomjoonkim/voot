import matplotlib as mpl
#mpl.use('Agg') # do this before importing plt to run with no DISPLAY
import matplotlib.pyplot as plt

from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.objectives import *
from keras import backend as K
from keras import initializers
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
import shutil
import os
import random 
import pandas as pd
from conveyor_belt_env import ConveyorBelt
from openravepy import *
from data_load_utils import format_RL_data

INFEASIBLE_SCORE = -sys.float_info.max
LAMBDA=0


def G_loss( dummy, pred ):
  return -K.mean(pred,axis=-1) # try to maximize the value of pred

def noise(n,z_size): 
  return np.random.normal(size=(n,z_size)).astype('float32')

def tile(x):
  reps = [1,1,32]
  return K.tile(x,reps)

class PlaceDDPG():
  def __init__(self,sess,dim_action,dim_state,tau,save_folder,explr_const,\
               key_configs=None,x_scaler=None,visualize=False):
    self.opt_G = Adam(lr=1e-4,beta_1=0.5)
    self.opt_D = Adam(lr=1e-3,beta_1=0.5)

    # initialize 
    self.initializer = initializers.glorot_normal()
    self.sess = sess

    # get setup dimensions for inputs
    self.dim_action = dim_action
    self.dim_state = dim_state
    self.n_key_confs = dim_state[0]
    self.key_configs = key_configs
    self.a_scaler = x_scaler

    self.v=visualize

    self.tau = tau
    
    # define inputs
    self.x_input = Input(shape=(dim_action,),name='x',dtype='float32')    # action
    self.w_input = Input( shape = dim_state,name='w',dtype='float32') # collision vector

    self.a_gen,self.disc,self.DG, = self.createGAN()
    self.save_folder = save_folder
    self.explr_const = explr_const

  def createGAN(self):
    disc = self.createDisc()
    a_gen,a_gen_output = self.createGen()
    for l in disc.layers:
      l.trainable=False
    DG_output = disc([a_gen_output,self.w_input]) 
    DG = Model(input=[self.w_input], output=[DG_output])
    DG.compile(loss={'disc_output':G_loss,},
              optimizer=self.opt_G,
              metrics=[])
    return a_gen,disc,DG

  def saveWeights(self,init=True,additional_name=''):
    self.a_gen.save_weights(self.save_folder+'/a_gen' +additional_name+ '.h5')
    self.disc.save_weights(self.save_folder+'/disc' +additional_name+ '.h5')

  def load_offline_weights(self,weight_f):
    self.a_gen.load_weights(self.save_folder+weight_f)

  def load_weights(self):
    best_rwd = -np.inf
    for weightf in os.listdir(self.save_folder):
      if weightf.find('a_gen')==-1: continue
      try:
        rwd = float(weightf.split('_')[-1][0:-3])
      except ValueError:
        continue
      if rwd > best_rwd:
        best_rwd  = rwd
        best_weight = weightf
    print "Using initial weight ",best_weight
    self.a_gen.load_weights(self.save_folder+'/'+best_weight)

  def resetWeights(self,init=True):
    if init:
      self.a_gen.load_weights('a_gen_init.h5')
      self.disc.load_weights('disc_init.h5')
    else:
      self.a_gen.load_weights(self.save_folder+'/a_gen.h5')
      self.disc.load_weights(self.save_folder+'/disc.h5')

  def createGen(self):
    init_ = self.initializer
    dropout_rate = 0.25
    dense_num = 64
    n_filters=64

    #K_H = self.k_input
    W_H = Reshape( (self.n_key_confs,self.dim_state[1],1) )(self.w_input)
    H = W_H
    H = Conv2D(filters=n_filters,\
               kernel_size=(1,self.dim_state[1]),\
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
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H1  = Conv2D(filters=n_filters,
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
    a_gen = Model(input=[self.w_input], output=a_gen_output)
    return a_gen,a_gen_output

  def createDisc(self):
    init_ = self.initializer
    dropout_rate = 0.25
    dense_num = 64
    n_filters=64

    #K_H = self.k_input
    X_H = RepeatVector(self.n_key_confs)(self.x_input)
    X_H = Reshape( (self.n_key_confs,self.dim_action,1))(X_H)
    W_H = Reshape( (self.n_key_confs,self.dim_state[1],1))(self.w_input)
    XK_H = Concatenate(axis=2)([X_H,W_H])

    H = Conv2D(filters=n_filters,\
               kernel_size=(1,self.dim_action+self.dim_state[1]),\
               strides=(1,1),
               activation='relu')(XK_H)
    H0 = H
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
    H1  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H1=H
    H = MaxPooling2D(pool_size=(2,1))(H1)
    H = Flatten()(H)
    H = Dense(dense_num,activation='relu')(H)
    H = Dense(dense_num,activation='relu')(H)

    disc_output = Dense(1, activation='linear',init=init_)(H)
    disc = Model(input=[self.x_input,self.w_input],\
                  output=disc_output,\
                  name='disc_output')
    disc.compile(loss='mse', optimizer=self.opt_D)
    return disc

  def predict(self,x,n_samples=1):
    x   = x.reshape((len(x),self.n_key_confs,self.dim_state[1]))
    dummy_sumA     = np.zeros((n_samples,1))
    dummy_old_pi_a = np.zeros((n_samples,self.dim_action))
 
    if n_samples==1:
      n = n_samples
      d = self.dim_action
      pred = self.a_gen.predict(x)
      noise = self.explr_const*np.random.randn(n,d)
      return self.a_scaler.inverse_transform( pred+noise )
    else:
      n = n_samples
      d = self.dim_action
      pred = self.a_gen.predict( np.tile(x,(n,1,1))  )
      noise = self.explr_const*np.random.randn(n,d)
      return self.a_scaler.inverse_transform( pred+noise )

  def soft_update(self,network,before,after):
    new_weights = network.get_weights()
    for i in range(len(before)):
      new_weights[i] = (1-self.tau)*before[i] +\
                            (self.tau)*after[i]
    network.set_weights(new_weights)

  def update_disc( self,batch_x,batch_w,batch_targets,batch_size ):
    before = self.disc.get_weights()
    checkpointer = ModelCheckpoint(filepath=self.save_folder+'/disc_weights.hdf5',\
                                   verbose=0,\
                                   save_best_only=True,\
                                   save_weights_only=True)
    self.disc.fit( {'x':batch_x,'w':batch_w},
                   batch_targets,
                   validation_split=0.1,
                   callbacks=[checkpointer],\
                   batch_size=batch_size,
                   epochs=1,
                   verbose=False )
    self.disc.load_weights(self.save_folder+'/disc_weights.hdf5')
    after = self.disc.get_weights()
    self.soft_update(self.disc,before,after)

  def update_pi( self,s_batch,batch_size ):
    # maximizes Q( pi(s_batch ) )
    y_labels = np.ones((len(s_batch),))  #dummy variable
    before = self.a_gen.get_weights()
    checkpointer = ModelCheckpoint(filepath=self.save_folder+'/DG_weights.hdf5',\
                                   verbose=False,\
                                   save_best_only=True,\
                                   save_weights_only=True)
    self.DG.fit( {'w':s_batch}, 
                 {'disc_output':y_labels,'a_gen_output':y_labels},  
                 callbacks=[checkpointer],\
                 validation_split=0.1,
                 batch_size=batch_size,
                 epochs=1, 
                 verbose=False )  
    self.DG.load_weights(self.save_folder+'/DG_weights.hdf5') # verfied that when I load DG weights, it loads a_gen weights
    after = self.a_gen.get_weights() # verfied that weights of disc does not change
    self.soft_update(self.a_gen,before,after)

  def train(self,states,actions,rewards,sprimes,\
            epochs=500,d_lr=1e-3,g_lr=1e-4):
    states=states.squeeze()
    sprimes = sprimes.squeeze()
    true_performance_list = []
    G_performance_list = []
    mse_list=[]

    n_data =states.shape[0]
    BATCH_SIZE = np.min([32,int(len(actions)*0.1)])
    if BATCH_SIZE==0:
      BATCH_SIZE = 1
    print BATCH_SIZE 

    K.set_value(self.opt_G.lr,g_lr)
    K.set_value(self.opt_D.lr,d_lr)
    print self.opt_G.get_config()

    current_best_J = -np.inf
    pfile = open(self.save_folder+'/performance.txt','w')
  
    # n_episodes = epochs*5
    # T = 20, but we update it once we finish executing all T 
    # This is because this is an episodic task - you can only learn meaningful moves
    # if you go deep in the trajectory.
    # So, we have 300*5*20 RL data
    for i in range(1,epochs):
      print 'Completed: %.2f%%'%(i/float(epochs)*100)
      stime=time.time()

      terminal_state_idxs = np.where(np.sum(np.sum(sprimes,axis=-1),axis=-1)==0)[0]
      nonterminal_mask = np.ones((sprimes.shape[0],1))
      nonterminal_mask[terminal_state_idxs,:] = 0

      # make the targets
      fake = self.a_gen.predict([sprimes])  # predicted by pi
      real = actions

      real_targets = rewards+np.multiply(self.disc.predict([fake,sprimes]),nonterminal_mask)
      stime = time.time()
      self.update_disc( real,states,real_targets,BATCH_SIZE )
      self.update_pi( states,BATCH_SIZE ) 
      print 'Fitting time',time.time() - stime
      
      # Technically speaking, we should update the policy every timestep.
      # What if we update it 100 times after we executed 5 episodes, each with 20 timesteps??
      stime=time.time()
      traj_list = []
      for n_iter in range(5):
        problem = ConveyorBelt() # different "initial" state 
        traj = problem.execute_policy(self,20,self.v)
        traj_list.append(traj)
        problem.env.Destroy()
        RaveDestroy()
      avg_J = np.mean([np.sum(traj['r']) for traj in traj_list])
      std_J = np.std([np.sum(traj['r']) for traj in traj_list])
      pfile = open(self.save_folder+'/performance.txt','a')
      pfile.write(str(i)+','+str(avg_J)+','+str(std_J)+'\n')
      pfile.close()
      print 'Score of this policy',avg_J

      # Add new data to the buffer - only if this was a non-zero trajectory
      if avg_J>1.0:
        new_s,new_a,new_r,new_sprime,new_sumR,_,new_traj_lengths = format_RL_data( traj_list )
        new_a = self.a_scaler.transform(new_a)
        states = np.r_[states,new_s.squeeze()]
        actions = np.r_[actions,new_a]
        rewards = np.r_[rewards,new_r]
        sprimes = np.r_[sprimes,new_sprime.squeeze()]
        print "Rollout time",time.time()-stime

      if avg_J > current_best_J:
        current_best_J = avg_J
        theta_star     = self.save_folder+'/policy_search_'+str(i)+'.h5'
        self.saveWeights(additional_name='tau_'+str(self.tau)+'epoch_'+\
                        str(i)+'_'+str(avg_J))  

