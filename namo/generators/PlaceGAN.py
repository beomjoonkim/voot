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

def make_save_folder(parent_dir,trial_name):
  assert( os.path.isdir(parent_dir) )
  save_folder_path = parent_dir+'/'+trial_name+'/'
  if not os.path.isdir(save_folder_path): 
    os.mkdir( save_folder_path )
  return save_folder_path
 
class PlaceGAN():
  def __init__(self,sess,dim_x,dim_w,dim_c,x_scaler,\
               parent_dir=None,\
               trial_name=None,\
               test_dir = None,\
               key_configs=None,\
               o_scaler=None,c0_scaler=None):
    is_test_mode = test_dir is not None
    if is_test_mode:
      self.save_folder = test_dir
    else:
      assert(parent_dir is not None)
      assert(trial_name is not None)
      self.save_folder = make_save_folder(parent_dir,trial_name)
    self.opt_G = Adam(lr=1e-4,beta_1=0.5)
    self.opt_D = Adam(lr=1e-3,beta_1=0.5) 

    # initialize 
    self.initializer = initializers.glorot_normal()
    self.sess = sess

    self.sess = sess
    self.dim_x = dim_x
    self.dim_w = dim_w
    self.dim_c = dim_c
    self.n_key_confs = dim_c[0]

    # x = action 
    # w = problem instance, or context 
    # c = collision vector
    self.x_input = Input(shape=(dim_x,),name='x',dtype='float32')  
    self.w_input = Input(shape=(dim_w,),name='w',dtype='float32')
    self.c_input = Input(shape=dim_c,name='c',dtype='float32')

    # define inputs
    self.dim_z = dim_x
    if dim_x <10:
      dim_z = dim_x
    else:
      dim_z = int(dim_x/2)
    self.dim_z = dim_z
    
    self.z = Input( shape = (self.dim_z,),name='z',dtype='float32')
  
    # define soap
    self.disc = self.createDisc()
    self.disc.summary()

    # following are needed whenever we wnat to use G
    self.key_configs = key_configs

    # scalers
    self.o_scaler = o_scaler
    self.c0_scaler = c0_scaler
    self.x_scaler  =x_scaler

    self.a_gen,self.disc,self.DG, = self.createGAN()
    self.a_gen.summary()
    self.disc.summary()

  def createGAN(self):
    disc = self.createDisc()
    a_gen,a_gen_output = self.createGen()
    for l in disc.layers:
      l.trainable=False
    DG_output = disc([a_gen_output,self.w_input,self.c_input]) 
    DG = Model(input=[self.z,self.w_input,self.c_input], output=[DG_output])
    DG.compile(loss='binary_crossentropy',
               optimizer=self.opt_G,
               metrics=[])
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
    dense_num = 64
    n_filters=64

    n_key_confs = self.n_key_confs
    C_H = Reshape( (self.n_key_confs,self.dim_c[1],1))(self.c_input)
    H = Conv2D(filters=n_filters,\
               kernel_size=(1,self.dim_c[1]),\
               strides=(1,1),
               activation='relu')(C_H)
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
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H = MaxPooling2D(pool_size=(2,1))(H)
    H = Flatten()(H)
    H = Dense(dense_num,activation='relu')(H)
    H = Dense(dense_num,activation='relu')(H)
    Z_H = Dense(dense_num,activation='relu')(self.z)
    H = Concatenate()([H,Z_H])
    a_gen_output = Dense(self.dim_x,
                         activation='linear',
                         init=init_,
                         name='a_gen_output')(H) 
    a_gen = Model(input=[self.z,self.w_input,self.c_input], output=a_gen_output)
    return a_gen,a_gen_output

  def createDisc(self):
    n_filters=64
    dense_num = 64
    init_ = self.initializer

    XW = Concatenate(axis=1)([self.x_input,self.w_input])
    XW = RepeatVector(self.n_key_confs)(XW)
    X_H = Reshape( (self.n_key_confs,self.dim_x+self.dim_w,1))(XW)
    C_H = Reshape( (self.n_key_confs,self.dim_c[1],1))(self.c_input)
    XC_H = Concatenate(axis=2)([X_H,C_H])


    H = Conv2D(filters=n_filters,\
               kernel_size=(1,self.dim_x+self.dim_w+self.dim_c[1]),\
               strides=(1,1),
               activation='relu')(XC_H)
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
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H = MaxPooling2D(pool_size=(2,1))(H)
    H = Flatten()(H)
    H = Dense(dense_num,activation='relu')(H)
    H = Dense(dense_num,activation='relu')(H)

    disc_output = Dense(1, activation='sigmoid',init=init_)(H)
    disc = Model(input=[self.x_input,self.w_input,self.c_input],\
                  output=[disc_output],\
                  name='disc_output')
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
    
  def train(self,a_data,w_data,c_data,score_data,\
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
    n_score_train = 5
    for i in range(1,epochs):
      stime=time.time()
      print 'Completed: %.2f%%'%(i/float(epochs)*100)
      for idx in range(0,a_data.shape[0],BATCH_SIZE):
        for score_train_idx in range(n_score_train):  
          # choose a batch of data
          indices = np.random.randint(0,a_data.shape[0],size=BATCH_SIZE)
          a_batch = np.array( a_data[indices,:] )
          w_batch = np.array( w_data[indices,:] )
          c_batch = np.array( c_data[indices,:] )

          # train \hat{S}
          # make fake and reals
          a_z = noise(BATCH_SIZE,self.dim_z)
          fake = self.a_gen.predict([a_z,w_batch,c_batch])
          real = a_batch

          # make their scores
          a_batch_ = np.vstack( [fake,real] )
          w_batch_ = np.vstack( [w_batch,w_batch] )
          c_batch_ = np.vstack( [c_batch,c_batch] )

          fake_labels = np.zeros((BATCH_SIZE,1))
          real_labels = np.ones((BATCH_SIZE,1))
          batch_labels = np.vstack( [fake_labels,real_labels] )

          self.disc.fit( {'x':a_batch_,'w':w_batch_,'c':c_batch_}, 
                         batch_labels,
                         epochs=1, 
                         verbose=False )

        # train G
        # why do i have labels for agen_output?
        a_z = noise(BATCH_SIZE,self.dim_z)
        y_labels = np.ones((BATCH_SIZE,))  #dummy variable
        #fake_before = self.a_gen.predict([a_z,w_batch,c_batch])
        self.DG.fit( {'z':a_z,'w':w_batch,'c':c_batch}, 
                     {'disc_output':y_labels},  
                     epochs = 1, 
                     verbose=0 )  
        #fake_after = self.a_gen.predict([a_z,w_batch,c_batch])
      
      # print D loss - apparently, low variance & decreasing implies it is working
      a_z = noise(a_data.shape[0],self.dim_z)

      Dtrue = np.mean(self.disc.predict([a_data,w_data,c_data]))

      fake = self.a_gen.predict([a_z,w_data,c_data])
      Dfake = np.mean(self.disc.predict([fake,w_data,c_data]))

      print np.mean(self.x_scaler.inverse_transform(self.a_gen.predict([a_z,w_data,c_data])),\
                    axis=0)


      print 'fake,real disc val = (%f,%f)'%(Dfake,Dtrue)
      mse = np.mean((Dtrue-score_data)**2)
      print 'mse = ' + str(mse)
      print 'adv loss = ', Dfake
      if i % 10 == 0:
        a_z = noise(a_data.shape[0],self.dim_z)
        Dtrue = np.mean(self.disc.predict([a_data,w_data,c_data]))
        fake = self.a_gen.predict([a_z,w_data,c_data ])
        Dfake = np.mean(self.disc.predict([fake,w_data,c_data]))
        self.saveWeights(additional_name='epoch_'+\
                          str(i)+'_'+str(epochs)+'_Dtrue_'+\
                          str(Dtrue)+'_Dfake_'+str(Dfake)
                          )
      print "Epoch took: %.2fs"%(time.time()-stime)


