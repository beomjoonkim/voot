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
from generators.Uniform import UniformPlace
from data_load_utils import get_sars_data
from sklearn.preprocessing import StandardScaler
from Q_loss_functions import *
from generators.DDPGPolicy import DDPGPolicy
from GAILPolicy import GAILPolicy,trpo_loss
    
class PickGAIL(GAILPolicy):
  def __init__(self,\
               sess,\
               dim_a,\
               dim_misc,\
               dim_cvec,\
               weight_dir,\
               key_configs,\
               Qloss,\
               d_lr,\
               g_lr,\
               n_score,architecture):
    super(PickGAIL,self).__init__(sess,\
                                  dim_a,\
                                  dim_misc,\
                                  dim_cvec,\
                                  weight_dir,\
                                  key_configs,\
                                  Qloss,\
                                  d_lr,\
                                  g_lr,\
                                  n_score,architecture)

  def createGen(self):
    init_ = self.initializer
    dense_num = 64
    n_filters = 64
    W    = RepeatVector(self.n_key_confs)(self.misc_input)
    W_H  = Reshape( (self.n_key_confs,self.dim_misc,1))(W)
    C_H  = Reshape( (self.n_key_confs,self.dim_cvec[-1],1))(self.c_input)
    WC_H = Concatenate(axis=2)([W_H,C_H])
    H = Conv2D(filters=n_filters,\
               kernel_size=(1,self.dim_cvec[1]+self.dim_misc),\
               strides=(1,1),
               activation='relu')(WC_H)
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
    a_gen_output = Dense(self.dim_a,
                         activation='linear',
                         init=init_,
                         name='a_gen_output')(H) 
    # these two are used for training purposes
    sumAweight_input = Input(shape=(1,),name='sumA',dtype='float32') 
    old_pi_a_input = Input(shape=(self.dim_a,),name='old_pi_a',dtype='float32')

    a_gen = Model( input=[self.misc_input,\
                          self.c_input,\
                          sumAweight_input,\
                          old_pi_a_input,\
                          self.tau_input],\
                  output=[a_gen_output] )
    # first loss:  pi/pi_old * A
    # second loss: KL(pi,pi_old)
    a_gen.compile( loss=trpo_loss(sumA_weight = sumAweight_input,\
                                  old_pi_a    = old_pi_a_input,\
                                  tau         = self.tau_input),\
                   optimizer=self.opt_G)
    return a_gen,a_gen_output

  def createV(self):
    dense_num = 64
    n_filters=64
    init_ = self.initializer

    W = self.misc_input
    W = RepeatVector(self.n_key_confs)(W)
    W_H = Reshape( (self.n_key_confs,self.dim_misc,1))(W)
    C_H = Reshape( (self.n_key_confs,self.dim_cvec[-1],1))(self.c_input)
    WC_H = Concatenate(axis=2)([W_H,C_H])
    H = Conv2D(filters=n_filters,\
               kernel_size=(1,self.dim_cvec[-1]+self.dim_misc),\
               strides=(1,1),
               activation='relu')(WC_H)
    H = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H = MaxPooling2D(pool_size=(2,1))(H)
    H = Flatten()(H)
    H = Dense(dense_num,activation='relu')(H)
    H = Dense(dense_num,activation='relu')(H)

    disc_output = Dense(1, activation='linear',init=init_)(H)
    V = Model(input=[self.misc_input,self.c_input],\
                  output=[disc_output],\
                  name='V_output')
    V.compile(loss='mse', optimizer=self.opt_V)
    return V


  def createDisc(self):
    dense_num = 64
    n_filters=64
    init_ = self.initializer

    XW = Concatenate(axis=1)([self.a_input,self.misc_input])
    XW = RepeatVector(self.n_key_confs)(XW)
    X_H = Reshape( (self.n_key_confs,self.dim_a+self.dim_misc,1))(XW)
    C_H = Reshape( (self.n_key_confs,self.dim_cvec[-1],1))(self.c_input)
    XC_H = Concatenate(axis=2)([X_H,C_H])
    H = Conv2D(filters=n_filters,\
               kernel_size=(1,self.dim_a+self.dim_cvec[-1]+self.dim_misc),\
               strides=(1,1),
               activation='relu')(XC_H)
    H = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H = MaxPooling2D(pool_size=(2,1))(H)
    H = Flatten()(H)
    H = Dense(dense_num,activation='relu')(H)
    H = Dense(dense_num,activation='relu')(H)
    disc_output = Dense(1, activation='sigmoid',init=init_)(H)
    disc = Model(input=[self.a_input,self.misc_input,self.c_input],\
                  output=disc_output,\
                  name='disc_output')
    disc.compile(loss='binary_crossentropy', optimizer=self.opt_D)
    return disc
  
