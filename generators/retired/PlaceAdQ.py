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

from multiprocessing import Process, Queue, Lock
from multiprocessing.dummy import Pool as ThreadPool 

INFEASIBLE_SCORE = -sys.float_info.max
LAMBDA=2

def augmented_mse( score_data, D_pred ):
  # Determine which of Dpred correspond to fake val  
  neg_mask      = tf.equal(score_data,INFEASIBLE_SCORE)
  y_neg         = tf.boolean_mask(D_pred,neg_mask) 
  
  # Determine which of Dpred correspond to true fcn val
  pos_mask      = tf.not_equal(score_data,INFEASIBLE_SCORE)
  y_pos         = tf.boolean_mask(D_pred,pos_mask) 
  score_pos     = tf.boolean_mask(score_data,pos_mask)

  # compute mse w.r.t true function values
  mse_on_true_data = K.mean( (K.square(score_pos - y_pos)), axis=-1)
  return mse_on_true_data+LAMBDA*K.mean( y_neg ) # try to minimize the value of y_neg

def G_loss( dummy, pred ):
  return -K.mean(pred,axis=-1) # try to maximize the value of pred

def noise(n,z_size): 
  return np.random.normal(size=(n,z_size)).astype('float32')

def tile(x):
  reps = [1,1,32]
  return K.tile(x,reps)

class PlaceAdQ():
  def __init__(self,sess,dim_action,dim_state,save_folder,tau,explr_const,\
               key_configs=None,x_scaler=None,c_scaler=None,visualize=False):
    self.opt_G = Adam(lr=1e-4,beta_1=0.5)
    self.opt_D = Adam(lr=1e-3,beta_1=0.5)

    # initialize 
    self.initializer = initializers.glorot_normal()
    self.sess = sess
    self.explr_const = explr_const

    # get setup dimensions for inputs
    self.dim_action = dim_action
    self.dim_state = dim_state
    self.n_key_confs = dim_state[0]
    self.key_configs = key_configs

    self.a_scaler = x_scaler
    
    # define inputs
    self.x_input = Input(shape=(dim_action,),name='x',dtype='float32')    # action
    self.w_input = Input( shape = dim_state,name='w',dtype='float32') # collision vector

    self.a_gen,self.disc,self.DG, = self.createGAN()
    self.save_folder = save_folder

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
    disc.compile(loss=augmented_mse, optimizer=self.opt_D)
    return disc

  def predict(self,x,n_samples=1):
    x   = x.reshape((len(x),self.n_key_confs,self.dim_state[1]))
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

  def rollout_thread(self,problem,i):
    with self.sess.graph.as_default():
      traj = problem.execute_policy(self,20)
    return traj

  def parallel_rollout(self):
    n_procs = 5
    pool = ThreadPool( n_procs )
    procs= []
    problems=[]
    for i in range(n_procs):
      problems.append(ConveyorBelt()) # different "initial" state 

    traj_list=[]
    for i in range(n_procs):
      print 'applying',i
      procs.append(pool.apply_async(self.rollout_thread,args=(problems[i],i,)))

    pool.close()
    pool.join()
    print [p.successful() for p in procs]
    for pidx,p in enumerate(procs):
      if not p.successful(): # Why does it ever fail? 
        print pidx,'Unsuccessful'
        traj_list.append( self.rollout_thread(problems[pidx],pidx))
      else:
        traj_list.append(p.get())

    return traj_list
   
  def train(self,states,actions,rewards,sprimes,\
            epochs=500,d_lr=1e-3,g_lr=1e-4):
    states=states.squeeze()
    sprimes = sprimes.squeeze()
    true_performance_list = []
    G_performance_list = []
    mse_list=[]

    n_data =states.shape[0]

    K.set_value(self.opt_G.lr,g_lr)
    K.set_value(self.opt_D.lr,d_lr)

    print self.opt_G.get_config()
    pfile = open(self.save_folder+'/performance.txt','w')
    pfile.close()

    current_best_J = -np.inf
    n_score_train = 1
    pfile = open(self.save_folder+'/performance.txt','w')
    for i in range(1,epochs):
      BATCH_SIZE = np.min([32,int(len(actions)*0.1)])
      if BATCH_SIZE==0:
        BATCH_SIZE = 1

      terminal_state_idxs = np.where(np.sum(np.sum(sprimes,axis=-1),axis=-1)==0)[0]
      nonterminal_mask = np.ones((sprimes.shape[0],1))
      nonterminal_mask[terminal_state_idxs,:] = 0

      stime=time.time()
      print 'Completed: %.2f%%'%(i/float(epochs)*100)
      n_iter = len(range(0,max(actions.shape[0],n_data),BATCH_SIZE))
      n_iter = min(100,n_iter)
      print "n_iter",n_iter
      #for idx in range(0,max(actions.shape[0],n_data),BATCH_SIZE):
      for _ in range(n_iter):
        for score_train_idx in range(n_score_train):
          # choose a batch of data - experience replay
          indices = np.random.randint(0,actions.shape[0],size=BATCH_SIZE)
          s_batch = np.array( states[indices,:] ) # collision vector
          a_batch = np.array( actions[indices,:] ) 
          r_batch = np.array( rewards[indices,:] )
          sprime_batch = np.array( sprimes[indices,:] )
          mask_batch   = np.array(nonterminal_mask[indices,:]) # 0 if terminal state, 1 ow
        
          fake = self.a_gen.predict([sprime_batch])
          real = a_batch

          # make their scores
          fake_targets = np.ones((BATCH_SIZE,1))*INFEASIBLE_SCORE # marks fake data
          real_targets = r_batch+np.multiply(self.disc.predict([fake,sprime_batch]),mask_batch)
          # Q = r(s,a)  if mask=0 if s is terminal


          batch_x = np.vstack( [fake,real] )
          batch_w = np.vstack( [s_batch,s_batch] )
          batch_targets = np.vstack( [fake_targets,real_targets] )
          self.disc.fit( {'x':batch_x,'w':batch_w},
                         batch_targets,
                         epochs=1, 
                         verbose=False )
       
        # train G
        y_labels = np.ones((BATCH_SIZE,))  #dummy variable
        self.DG.fit( {'w':s_batch}, 
                     {'disc_output':y_labels,'a_gen_output':y_labels},  
                     epochs=1, 
                     verbose=0 )  
      print "Training took: %.2fs"%(time.time()-stime)
      # Try policy - 5 trajectories, each 20 long
      """
      traj_list = []
      for n_iter in range(5):
        problem = ConveyorBelt() # different "initial" state 
        traj = problem.execute_policy(self,20)
        traj_list.append(traj)
        problem.env.Destroy()
        RaveDestroy()
      """
      stime=time.time()
      traj_list = self.parallel_rollout()
      print "Rollout took: %.2fs"%(time.time()-stime)
      avg_J = np.mean([np.sum(traj['r']) for traj in traj_list])
      std_J = np.std([np.sum(traj['r']) for traj in traj_list])
      pfile = open(self.save_folder+'/performance.txt','a')
      pfile.write(str(i)+','+str(avg_J)+','+str(std_J)+'\n')
      pfile.close()
      print 'Score of this policy',avg_J

      # Add new data to the buffer
      new_s,new_a,new_r,new_sprime,new_sumR,_,new_traj_lengths = format_RL_data(traj_list)
      new_a = self.a_scaler.transform(new_a)
      states = np.r_[states,new_s.squeeze()]
      actions = np.r_[actions,new_a]
      rewards = np.r_[rewards,new_r]
      sprimes = np.r_[sprimes,new_sprime.squeeze()]

      if avg_J > current_best_J:
        current_best_J = avg_J
        theta_star     = self.save_folder+'/policy_search_'+str(i)+'.h5'
        self.saveWeights(additional_name='lambda_'+str(LAMBDA)+'epoch_'+\
                        str(i)+'_'+str(avg_J))  

      print "Epoch took: %.2fs"%(time.time()-stime)


