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

INFEASIBLE_SCORE = -sys.float_info.max

def tau_loss( tau ):
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
    return mse_on_true_data+tau[0]*K.mean( y_neg ) # try to minimize the value of y_neg
  return augmented_mse

def G_loss( dummy, pred ):
  return -K.mean(pred,axis=-1) # try to maximize the value of pred

def noise(n,z_size): 
  return np.random.normal(size=(n,z_size)).astype('float32')

def tile(x):
  reps = [1,1,32]
  return K.tile(x,reps)

class PlaceSOAP():
  def __init__(self,sess,dim_data,dim_context,save_folder,tau,explr_const,\
               key_configs=None,x_scaler=None,c_scaler=None,visualize=False):
    self.opt_G = Adam(lr=1e-4,beta_1=0.5)
    self.opt_D = Adam(lr=1e-3,beta_1=0.5)

    # initialize 
    self.initializer = initializers.glorot_normal()
    self.sess = sess

    # get setup dimensions for inputs
    self.dim_data = dim_data
    self.dim_context = dim_context
    self.n_key_confs = dim_context[0]
    self.key_configs = key_configs

    self.x_scaler = x_scaler
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

    self.a_gen,self.disc,self.DG, = self.createGAN()
    self.save_folder = save_folder
    self.tau = tau

  def createGAN(self):
    disc = self.createDisc()
    a_gen,a_gen_output = self.createGen()
    for l in disc.layers:
      l.trainable=False
    DG_output = disc([a_gen_output,self.w_input,self.w_input]) 
    DG = Model(input=[self.z,self.w_input], output=[DG_output])
    DG.compile(loss={'disc_output':G_loss,},
              optimizer=self.opt_G,
              metrics=[])
    return a_gen,disc,DG

  def saveWeights(self,additional_name=''):
    self.a_gen.save_weights(self.save_folder+'/a_gen' +additional_name+ '.h5')
    self.disc.save_weights(self.save_folder+'/disc' +additional_name+ '.h5')

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
    Z_H = Dense(dense_num,activation='relu')(self.z)
    H = Concatenate()([H,Z_H])
    a_gen_output = Dense(self.dim_data,
                         activation='linear',
                         init=init_,
                         name='a_gen_output')(H) 
    a_gen = Model(input=[self.z,self.w_input], output=a_gen_output)
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

    disc_output = Dense(1, activation='linear',init=init_)(H)
    disc = Model(input=[self.x_input,self.w_input,self.tau_input],\
                  output=disc_output,\
                  name='disc_output')
    disc.compile(loss=tau_loss(self.tau_input), optimizer=self.opt_D)
    return disc

  def generate(self,w,n_samples): 
    if w.shape[0]==1 and n_samples==1:
      a_z = noise(1,self.dim_data)      
      g   = self.a_gen.predict([a_z,w])
    elif w.shape[0]==1 and n_samples>1:
      a_z = noise(n_samples,self.dim_data)
      w   = np.tile(w,(n_samples,1))
      g   = self.a_gen.predict([a_z,w])
    elif w.shape[0]>1 and n_samples==1:
      a_z = noise(w.shape[0],self.dim_data)
      g   = self.a_gen.predict([a_z,w])
    else:
      return None
      print "cannot handle this case"
    return self.x_scaler.inverse_transform(self.a_gen.predict([a_z,w]))

  def predict(self,w,n_samples=1):    
    if w.shape[0]==1 and n_samples>1:
      a_z = noise(n_samples,self.dim_data)
      w   = np.tile(w,(n_samples,1))
      w   = w.reshape((n_samples,self.n_key_confs,self.dim_context[1]))
      g   = self.x_scaler.inverse_transform(self.a_gen.predict([a_z,w]))
    elif w.shape[0]==1 and n_samples==1:
      a_z = noise(w.shape[0],self.dim_data)
      w   = w.reshape((1,self.n_key_confs,self.dim_context[1]))
      g   = self.x_scaler.inverse_transform(self.a_gen.predict([a_z,w]))
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

   
  def train(self,states,actions,sumRs,dummy,\
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

    self.tau = np.tile(self.tau,(BATCH_SIZE*2,1))

    K.set_value(self.opt_G.lr,g_lr)
    K.set_value(self.opt_D.lr,d_lr)

    print self.opt_G.get_config()

    current_best_J = -np.inf
    n_score_train = 1
    performance_list = []
    pfile = open(self.save_folder+'/performance.txt','w')
    for i in range(1,epochs):
      stime=time.time()
      print 'Completed: %.2f%%'%(i/float(epochs)*100)
      for idx in range(0,actions.shape[0],BATCH_SIZE):
        for score_train_idx in range(n_score_train):
          # choose a batch of data
          indices    = np.random.randint(0,actions.shape[0],size=BATCH_SIZE)
          s_batch    = np.array( states[indices,:] ) # collision vector
          a_batch    = np.array( actions[indices,:] ) 
          sumR_batch = np.array( sumRs[indices,:] )
        
          # train \hat{S}
          # make fake and reals
          a_z = noise(BATCH_SIZE,self.dim_z)
          fake = self.a_gen.predict([a_z,s_batch])
          real = a_batch

          # make their scores
          fake_scores = np.ones((BATCH_SIZE,1))*INFEASIBLE_SCORE # marks fake data
          real_scores = sumR_batch

          batch_x = np.vstack( [fake,real] )
          batch_w = np.vstack( [s_batch,s_batch] )
          batch_scores = np.vstack( [fake_scores,real_scores] )
          self.disc.fit( {'x':batch_x,'w':batch_w,'tau':self.tau},
                         batch_scores,
                         epochs=1, 
                         verbose=False )
       
        # train G
        # why do i have labels for agen_output?
        a_z = noise(BATCH_SIZE,self.dim_z)
        y_labels = np.ones((BATCH_SIZE,))  #dummy variable
        self.DG.fit( {'z':a_z,'w':s_batch}, 
                     {'disc_output':y_labels,'a_gen_output':y_labels},  
                     epochs = 1, 
                     verbose=0 )  
      self.saveWeights(additional_name='_epoch_'+ str(i))
      print "Epoch took: %.2fs"%(time.time()-stime)


