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


INFEASIBLE_SCORE = -sys.float_info.max

def noise(n,z_size): 
  return np.random.normal(size=(n,z_size)).astype('float32')

def tile(x):
  reps = [1,1,32]
  return K.tile(x,reps)

class PlaceGAN():
  def __init__(self,sess,dim_data,dim_context,dim_k,save_folder,\
               key_configs=None,k_data=None,x_scaler=None,c_scaler=None):
    self.opt_G = Adam(lr=1e-4,beta_1=0.5)
    self.opt_D = Adam(lr=1e-3,beta_1=0.5)

    # initialize 
    self.initializer = initializers.glorot_normal()
    self.sess = sess

    # get setup dimensions for inputs
    self.dim_data = dim_data
    self.dim_context = dim_context
    self.dim_k = dim_k
    self.n_key_confs = dim_context[0]
    self.k_data = k_data
    self.key_configs =key_configs

    self.x_scaler = x_scaler
    self.c_scaler = c_scaler
    
    # define inputs
    self.x_input = Input(shape=(dim_data,),name='x',dtype='float32')  
    self.w_input = Input( shape = dim_context,name='w',dtype='float32')
    self.k_input = Input( shape = dim_k,name='k',dtype='float32')
 

    if dim_data <10:
      dim_z = dim_data
    else:
      dim_z = int(dim_data/2)
    self.dim_z=dim_z
    self.z = Input( shape = (self.dim_z,),name='z',dtype='float32')

    self.a_gen,self.disc,self.DG, = self.createGAN()
    self.a_gen.summary()
    self.disc.summary()
    self.save_folder = save_folder


  def createGAN(self):
    disc = self.createDisc()
    a_gen,a_gen_output = self.createGen()
    for l in disc.layers:
      l.trainable=False
    DG_output = disc([a_gen_output,self.w_input]) 
    DG = Model(input=[self.z,self.w_input], output=[DG_output])
    DG.compile(loss='binary_crossentropy',optimizer=self.opt_G,metrics=[])
    return a_gen,disc,DG

  def saveWeights(self,init=True,additional_name=''):
    self.a_gen.save_weights(self.save_folder+'/a_gen' +additional_name+ '.h5')
    self.disc.save_weights(self.save_folder+'/disc'+additional_name+'.h5')

  def load_offline_weights(self,weight_f):
    self.a_gen.load_weights(self.save_folder+weight_f)

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

    K_H = self.k_input
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
    self.Z_H = Model(input=[self.z], output=Z_H)
    self.H1_gen = Model(input=[self.z,self.w_input], output=H1)
    return a_gen,a_gen_output



  def createDisc(self):
    init_ = self.initializer
    dropout_rate = 0.25
    dense_num = 64
    n_filters=64

    K_H = self.k_input
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
    #self.H0= Model(input=[self.x_input,self.k_input,self.w_input], output=H0)
    #self.H1= Model(input=[self.x_input,self.w_input], output=H1)
    disc.compile(loss='binary_crossentropy', optimizer=self.opt_D)
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
      import pdb;pdb.set_trace()
      g   = self.a_gen.predict([a_z,w])
    else:
      return None
      print "cannot handle this case"

    return self.a_gen.predict([a_z,w])
    
  def train(self,w_data,k_data,a_data,\
            epochs=500,d_lr=1e-3,g_lr=1e-4):
    true_performance_list = []
    G_performance_list = []
    mse_list=[]

    n_data =w_data.shape[0]
    BATCH_SIZE = np.min([32,int(len(a_data)*0.1)])
    if BATCH_SIZE==0:
      BATCH_SIZE = 1
    print BATCH_SIZE 

    K.set_value(self.opt_G.lr,g_lr)
    K.set_value(self.opt_D.lr,d_lr)

    print self.opt_G.get_config()
    n_score_train = 1
    for i in range(1,epochs):
      print 'Completed: %.2f%%'%(i/float(epochs)*100)
      for idx in range(0,a_data.shape[0],BATCH_SIZE):
        # choose a batch of data
        indices = np.random.randint(0,a_data.shape[0],size=BATCH_SIZE)
        context_batch = np.array( w_data[indices,:] )
        action_batch  = np.array( a_data[indices,:] )
        batch_konf    = np.array( k_data[indices,:] )
      
        # train \hat{S}
        # make fake and reals
        a_z = noise(BATCH_SIZE,self.dim_z)
        fake = self.a_gen.predict([a_z,context_batch])
        real = action_batch

        # make their labels
        fake_labels = np.zeros((BATCH_SIZE,1))
        real_labels = np.ones((BATCH_SIZE,1))
        
        # organize them
        batch_x = np.vstack( [fake,real] )
        batch_w = np.vstack( [context_batch,context_batch] )
        batch_k = np.vstack( [batch_konf,batch_konf] )
        batch_labels = np.vstack( [fake_labels,real_labels] )

        self.disc.fit( {'x':batch_x,'w':batch_w},
                       batch_labels,
                       epochs=1, 
                       verbose=False )
       
        # train G
        a_z = noise(BATCH_SIZE,self.dim_z)
        y_labels = np.ones((BATCH_SIZE,))  #dummy variable
        self.DG.fit( {'z':a_z,'w':context_batch,'k':batch_konf}, 
                     {'disc_output':y_labels}, 
                     epochs = 1, 
                     verbose=0 )  
      
      # print D loss - apparently, low variance & decreasing implies it is working
      a_z = noise(a_data.shape[0],self.dim_z)
      Dtrue = np.mean(self.disc.predict([a_data,w_data]))
      Dfake = np.mean(self.disc.predict([self.a_gen.predict([a_z,w_data]),\
                                w_data]))
      print 'fake,real disc val = (%f,%f)'%(Dfake,Dtrue)
      print 'adv loss = ', Dfake

      if i % 10 == 0:
        self.saveWeights(additional_name='epoch_'+\
                          str(i)+'_'+str(epochs)+'_Dtrue_'+\
                          str(Dtrue)+'_Dfake_'+str(Dfake)
                          )


