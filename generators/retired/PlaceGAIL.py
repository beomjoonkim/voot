import matplotlib as mpl
#mpl.use('Agg') # do this before importing plt to run with no DISPLAY
import matplotlib.pyplot as plt

from keras.layers import *
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
from keras.callbacks import ModelCheckpoint



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
from PlaceEvaluator import PlaceEvaluator
from conveyor_belt_env import ConveyorBelt
from openravepy import *
from data_load_utils import format_RL_data
import scipy.signal
from PlaceTRPO import discount




INFEASIBLE_SCORE = -sys.float_info.max
def trpo_loss(sumA_weight,tau):
  def loss( actions, pi_pred ):
    p_new     = K.exp(-K.sum(K.square(pi_pred - actions),axis=-1,keepdims=True) )
    L_cpi     = tf.multiply(sumA_weight, p_new)
    clipped   = tf.clip_by_value(p_new,1-tau[0,0],1+tau[0,0])
    L_clipped = tf.multiply(sumA_weight,clipped)
    L = tf.minimum(L_cpi,L_clipped) 
    return -L
  return loss

def noise(n,z_size): 
  return np.random.normal(size=(n,z_size)).astype('float32')

def tile(x):
  reps = [1,1,32]
  return K.tile(x,reps)

class PlaceGAIL():
  def __init__(self,sess,dim_data,dim_context,save_folder,tau,explr_const,\
               key_configs=None,a_scaler=None,c_scaler=None,visualize=False):
    self.opt_G = Adam(lr=1e-4,beta_1=0.5)
    self.opt_D = Adam(lr=1e-3,beta_1=0.5)
    self.opt_V = Adam(lr=1e-3,beta_1=0.5)

    # initialize 
    self.initializer = initializers.glorot_normal()
    self.sess = sess

    # get setup dimensions for inputs
    self.dim_data = dim_data
    self.dim_context = dim_context
    self.n_key_confs = dim_context[0]
    self.key_configs = key_configs

    self.a_scaler = a_scaler
    self.c_scaler = c_scaler
    
    # define inputs
    self.x_input = Input(shape=(dim_data,),name='x',dtype='float32')    # action
    self.w_input = Input( shape = dim_context,name='w',dtype='float32') # collision vector
    self.tau_input = Input( shape = (1,),name='tau',dtype='float32') # collision vector

    self.v=visualize
    self.explr_const = explr_const

    if dim_data <10:
      dim_z = dim_data
    else:
      dim_z = int(dim_data/2)
    self.dim_z=dim_z
    self.z = Input( shape = (self.dim_z,),name='z',dtype='float32')

    self.a_gen,self.disc,self.Vfcn = self.createGAN()
    self.save_folder = save_folder
    self.tau = tau

  def createGAN(self):
    disc = self.createDisc()
    a_gen,a_gen_output = self.createGen()
    V = self.createV()
    return a_gen,disc,V

  def saveWeights(self,additional_name=''):
    self.a_gen.save_weights(self.save_folder+'/a_gen' +additional_name+ '.h5')
    self.disc.save_weights(self.save_folder+'/disc' +additional_name+ '.h5')
    self.Vfcn.save_weights(self.save_folder+'/Vfcn' +additional_name+ '.h5')

  def load_weights(self,weight_f):
    print "Loading the weight ",weight_f
    self.a_gen.load_weights( weight_f)


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
    W_H = Reshape( (self.n_key_confs,self.dim_context[1],1) )(self.w_input)
    H = W_H
    H = Conv2D(filters=n_filters,\
               kernel_size=(1,self.dim_context[1]),\
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
    a_gen_output = Dense(self.dim_data,
                         activation='linear',
                         init=init_,
                         name='a_gen_output')(H) 
    sumAweight_input = Input(shape=(1,),name='sumA',dtype='float32') 
    a_gen = Model(input=[self.w_input,\
                         sumAweight_input,\
                         self.tau_input],\
                  output=a_gen_output)
    a_gen.compile(loss=trpo_loss(sumAweight_input,\
                                 self.tau_input),\
                  optimizer=self.opt_G)
    return a_gen,a_gen_output

  def createDisc(self):
    init_ = self.initializer
    dropout_rate = 0.25
    dense_num = 64
    n_filters=64

    #K_H = self.k_input
    X_H = RepeatVector(self.n_key_confs)(self.x_input)
    X_H = Reshape( (self.n_key_confs,self.dim_data,1))(X_H)
    W_H = Reshape( (self.n_key_confs,self.dim_context[1],1))(self.w_input)
    XK_H = Concatenate(axis=2)([X_H,W_H])

    H = Conv2D(filters=n_filters,\
               kernel_size=(1,self.dim_data+self.dim_context[1]),\
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
    disc_output = Dense(1, activation='sigmoid',init=init_)(H)
    disc = Model(input=[self.x_input,self.w_input],\
                  output=disc_output,\
                  name='disc_output')
    disc.compile(loss='binary_crossentropy', optimizer=self.opt_D)
    return disc

  def createV(self):
    init_ = self.initializer
    dropout_rate = 0.25
    dense_num = 64
    n_filters=64

    #K_H = self.k_input
    W_H = Reshape( (self.n_key_confs,self.dim_context[1],1))(self.w_input)
    H = Conv2D(filters=n_filters,\
               kernel_size=(1,self.dim_context[1]),\
               strides=(1,1),
               activation='relu')(W_H)
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
    V = Model(input=[self.w_input],\
                  output=disc_output,\
                  name='disc_output')
    V.compile(loss='mse', optimizer=self.opt_V)
    return V

  def predict(self,w,n_samples=1):    
    dummy_sumA     = np.zeros((n_samples,1))
    tau = np.tile(self.tau,(n_samples,1))
    if w.shape[0]==1 and n_samples>1:
      w     = np.tile(w,(n_samples,1))
      w     = w.reshape((n_samples,self.n_key_confs,self.dim_context[1]))
      noise = self.explr_const*np.random.randn(n_samples,self.dim_data)
    elif w.shape[0]==1 and n_samples==1:
      w     = w.reshape((1,self.n_key_confs,self.dim_context[1]))
      noise = self.explr_const*np.random.randn(n_samples,self.dim_data)
    pred = self.a_gen.predict([w,dummy_sumA,tau])
    g = self.a_scaler.inverse_transform( pred + noise )
    return g

  def evaluate(self,visualize=False):
    # Try the policy
    traj_list = []
    for n_iter in range(5):
      problem = ConveyorBelt() # different "initial" state 
      traj = problem.execute_policy(self,20,self.v)
      traj_list.append(traj)
      problem.env.Destroy()
      RaveDestroy()
    avg_J = np.mean([np.sum(traj['r']) for traj in traj_list])
    std_J = np.std([np.sum(traj['r']) for traj in traj_list])
    return avg_J,std_J

  def compute_r_using_D(self,traj_list):
    new_r = []
    for traj in traj_list:
      states  = np.array(traj['s'])
      states  = states.reshape( (len(states),self.n_key_confs,2) )
      actions = traj['a'] 
      actions = np.array(actions).reshape((len(actions),3))
      actions = self.a_scaler.transform(actions)
      new_r.append( np.log(self.disc.predict([ actions,states])) )
    new_sumR = []
    for traj_r in new_r:
      new_sumR.append( np.array([np.sum(traj_r[i:]) for i in range(len(traj_r))]) )
    return np.vstack(new_r).squeeze(),np.hstack(new_sumR)

  def update_V(self,states,sumR):
    n_data =states.shape[0]
    batch_size = np.min([32,int(len(states)*0.1)])
    if batch_size==0:
      batch_size = 1
    
    checkpointer = ModelCheckpoint(filepath=self.save_folder+'/weights.hdf5',\
                                   verbose=0,\
                                   save_best_only=True,\
                                   save_weights_only=True)
    print "Fitting V"
    self.Vfcn.fit( states,sumR,epochs=20,validation_split=0.1, \
                   callbacks=[checkpointer],batch_size=batch_size,verbose=False ) 
    self.Vfcn.load_weights(self.save_folder+'/weights.hdf5')

  def update_pi(self,states,actions,sumA):
    n_data =states.shape[0]
    batch_size = np.min([32,int(len(actions)*0.1)])
    if batch_size==0:
      batch_size = 1
    checkpointer = ModelCheckpoint(filepath=self.save_folder+'/pi_weights.hdf5',\
                                   verbose=0,\
                                   save_best_only=True,\
                                   save_weights_only=True)
    print "Fitting pi..."
    tau = np.tile(self.tau,(n_data,1))
    self.a_gen.fit( [states,sumA,tau],\
                    actions,epochs=20,validation_split=0.1,\
                    batch_size=batch_size,\
                    callbacks=[checkpointer],\
                    verbose=False )  
    print "Done!"
    self.a_gen.load_weights(self.save_folder+'/pi_weights.hdf5')

  def compute_A( self,states,actions,sprimes,rewards,traj_lengths ):
    Vsprime = np.array([self.Vfcn.predict(s[None,:])[0,0] \
                        if np.sum(s)!=0 else 0 for s in sprimes])
    V = self.Vfcn.predict(states)
    n_data =len(Vsprime)
    Vsprime = Vsprime.reshape( (n_data,1) )
    rewards = rewards.reshape((n_data,1) )  
    V       = V.reshape((n_data,1))
    
    Q =  rewards + Vsprime
    A = Q - V
    sumA=[]
    for i in range(len(A)):
      try:
        sumA.append( discount(A[i:i+traj_lengths[i]],1) )
      except IndexError:
        break

    Astd = A.std()
    normalizedA = (A - A.mean()) / Astd if not np.isclose(Astd,0) else A
    return normalizedA


  def train(self,states,actions,\
            epochs=500,d_lr=1e-3,g_lr=1e-4):
    states=states.squeeze()

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
    n_score_train = 1
    performance_list = []
    pfile = open(self.save_folder+'/performance.txt','w')
    for i in range(1,epochs):
      stime=time.time()

      # Rollouts
      # 5 trajectories, each 20 long
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
      pfile.write(str(avg_J)+','+str(std_J)+'\n')
      pfile.close()
      print 'Score of this policy',avg_J
  
      # new rollout dataset
      new_s,new_a,new_r,new_sprime,new_sumR,_,new_traj_lengths = format_RL_data( traj_list )
      new_a = self.a_scaler.transform(new_a)

      # choose a batch of data
      indices    = np.random.randint(0,actions.shape[0],size=BATCH_SIZE)
      s_batch    = np.array( states[indices,:] ) # collision vector
      a_batch    = np.array( actions[indices,:] ) 

      pi_indices    = np.random.randint(0,new_a.shape[0],size=BATCH_SIZE)
      pi_s_batch    = np.array( new_s[pi_indices,:] ) # collision vector
      pi_a_batch    = np.array( new_a[pi_indices,:] ) 
     
      # make their scores
      fake_scores = np.zeros((BATCH_SIZE,1))
      real_scores = np.ones((BATCH_SIZE,1))
      batch_x = np.vstack( [pi_a_batch,a_batch] )
      batch_w = np.vstack( [pi_s_batch,s_batch] )
      batch_scores = np.vstack( [fake_scores,real_scores] )

      # Update  D
      self.disc.fit( {'x':batch_x,'w':batch_w},
                     batch_scores,
                     epochs=1, 
                     verbose=False )
      new_r,new_sumR = self.compute_r_using_D( traj_list )

      # update value function
      self.update_V(new_s,new_sumR)

      # update policy
      new_sumA = self.compute_A(new_s,new_a,new_sprime,new_r,new_traj_lengths)
      self.update_pi(new_s,new_a,new_sumA)

      self.saveWeights(additional_name='epoch_'+\
                      str(i)+'_'+str(avg_J))  

      print 'Completed: %.2f%%'%(i/float(epochs)*100)
      print "Epoch took: %.2fs"%(time.time()-stime)


