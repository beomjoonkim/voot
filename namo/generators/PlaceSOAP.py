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
import math
import csv
import shutil
import os
import random 
import pickle
import pandas as pd

from NAMO_env import NAMO
from data_load_utils import get_sars_data
from Q_loss_functions import *
from Policy import Policy,noise
from slice_functions import *

class PlaceSOAP(Policy):
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
               n_score,
               explr_const,\
               tau,architecture):
    super(PlaceSOAP,self).__init__(sess,\
                                   dim_a,\
                                   dim_misc,\
                                   dim_cvec,\
                                   weight_dir,\
                                   key_configs,\
                                   Qloss,\
                                   d_lr,\
                                   g_lr,\
                                   n_score,\
                                   explr_const,\
                                   tau,architecture)
  def createGen4(self):  
    init_ = self.initializer
    dense_num = 64
    n_filters=64
    n_key_confs = self.n_key_confs

    self.dim_misc = 3 # first three, which is the current config
    W_H    = Lambda(slice_c0)(self.misc_input)
    W_H  = RepeatVector(self.n_key_confs)(W_H)
    W_H  = Reshape( (self.n_key_confs,3,1))(W_H)

    C_H = Reshape( (self.n_key_confs,self.dim_cvec[-1],1))(self.c_input)
    WC_H = Concatenate(axis=2)([W_H,C_H])

    H = Conv2D(filters=n_filters,\
               kernel_size=(1,4+3),\
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
    #Z_H = Dense(dense_num,activation='relu')(self.z)
    #H = Concatenate()([H,Z_H])
    a_gen_output = Dense(self.dim_a,
                         activation='linear',
                         init=init_,
                         name='a_gen_output')(H) 
    a_gen = Model(input=[self.z,self.misc_input,self.c_input], output=a_gen_output)
    self.preproc = Model(input=[self.misc_input,self.c_input],\
                  output=[W_H],\
                  name='preproc')
    return a_gen,a_gen_output

  def createDisc4(self):
    n_filters=64
    dense_num = 64
    init_ = self.initializer

    self.dim_misc = 3 # first three, which is the current config
    shared_l1 = Dense(dense_num,activation='relu')

    #A_H  = shared_l1(self.a_input)
    A_H=self.a_input
    A_H  = RepeatVector(self.n_key_confs)(A_H)
    A_H  = Reshape( (self.n_key_confs,3,1))(A_H)

    W    = Lambda(slice_c0)(self.misc_input)
    #W_H  = shared_l1(W)
    W_H  = W
    W_H  = RepeatVector(self.n_key_confs)(W_H)
    W_H  = Reshape( (self.n_key_confs,3,1))(W_H)

    AW = Concatenate(axis=2)([A_H,W_H])
    C_H = Reshape( (self.n_key_confs,self.dim_cvec[-1],1))(self.c_input)
    AC_H = Concatenate(axis=2)([AW,C_H])
    H = Conv2D(filters=n_filters,\
               kernel_size=(1,4+3*2),\
               strides=(1,1),
               activation='relu')(AC_H)
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
    disc = Model(input=[self.a_input,self.misc_input,self.c_input,self.tau_input],\
                  output=[disc_output],\
                  name='disc_output')
    disc.compile(loss=tau_loss(self.tau_input), optimizer=self.opt_D)
    return disc

  def createGen2(self):  
    init_ = self.initializer
    dense_num = 64
    n_filters=64
    n_key_confs = self.n_key_confs

    self.dim_misc = 3 # first three, which is the current config

    W   = Lambda(slice_c0)(self.misc_input)
    W_H = Dense(dense_num,activation='relu')(W)
    W_H = Dense(self.n_key_confs,activation='relu')(W_H)
    W_H = Reshape( (self.n_key_confs,1,1) )(W_H)

    C_H = Reshape( (self.n_key_confs,self.dim_cvec[-1],1))(self.c_input)
    WC_H = Concatenate(axis=2)([W_H,C_H])

    H = Conv2D(filters=n_filters,\
               kernel_size=(1,2+2+1),\
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
    a_gen = Model(input=[self.z,self.misc_input,self.c_input], output=a_gen_output)
    return a_gen,a_gen_output

  def createDisc2(self):
    n_filters=64
    dense_num = 64
    init_ = self.initializer


    self.dim_misc = 3 # first three, which is the current config

    A_H = Dense(dense_num,activation='relu')(self.a_input)
    A_H = Dense(self.n_key_confs,activation='relu')(A_H)
    A_H = Reshape( (self.n_key_confs,1,1) )(A_H)

    W   = Lambda(slice_c0)(self.misc_input)
    W_H = Dense(dense_num,activation='relu')(W)
    W_H = Dense(self.n_key_confs,activation='relu')(W_H)
    W_H = Reshape( (self.n_key_confs,1,1) )(W_H)

    AW = Concatenate(axis=2)([A_H,W_H])
    C_H = Reshape( (self.n_key_confs,self.dim_cvec[-1],1))(self.c_input)
    AC_H = Concatenate(axis=2)([AW,C_H])
    H = Conv2D(filters=n_filters,\
               kernel_size=(1,2+2+2),\
               strides=(1,1),
               activation='relu')(AC_H)
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
    disc = Model(input=[self.a_input,self.misc_input,self.c_input,self.tau_input],\
                  output=[disc_output],\
                  name='disc_output')
    disc.compile(loss=tau_loss(self.tau_input), optimizer=self.opt_D)
    return disc

  def createDisc1(self):
    n_filters=64
    dense_num = 64
    init_ = self.initializer


    self.dim_misc = 3 # first three, which is the current config

    shared_l1 = Dense(dense_num,activation='relu')
    shared_l2 = Dense(self.n_key_confs,activation='relu')
    A_H = shared_l2(shared_l1(self.a_input))
    A_H = Reshape( (self.n_key_confs,1,1) )(A_H)

    W   = Lambda(slice_c0)(self.misc_input)
    W_H = shared_l2(shared_l1(W))
    W_H = Reshape( (self.n_key_confs,1,1) )(W_H)

    AW = Concatenate(axis=2)([A_H,W_H])

    C_H = Reshape( (self.n_key_confs,self.dim_cvec[-1],1))(self.c_input)
    AC_H = Concatenate(axis=2)([AW,C_H])
    H = Conv2D(filters=n_filters,\
               kernel_size=(1,2+2+2),\
               strides=(1,1),
               activation='relu')(AC_H)
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
    disc = Model(input=[self.a_input,self.misc_input,self.c_input],\
                  output=[disc_output],\
                  name='disc_output')
    disc.compile(loss=adv_mse, optimizer=self.opt_D)
    return disc

  def createGen1(self):  
    init_ = self.initializer
    dense_num = 64
    n_filters=64
    n_key_confs = self.n_key_confs

    self.dim_misc = 3 # first three, which is the current config

    shared_l1 = Dense(dense_num,activation='relu')
    shared_l2 = Dense(self.n_key_confs,activation='relu')

    W   = Lambda(slice_c0)(self.misc_input)
    W_H = shared_l2(shared_l1(W))
    W_H = Reshape( (self.n_key_confs,1,1) )(W_H)

    C_H = Reshape( (self.n_key_confs,self.dim_cvec[-1],1))(self.c_input)
    WC_H = Concatenate(axis=2)([W_H,C_H])

    H = Conv2D(filters=n_filters,\
               kernel_size=(1,2+2+1),\
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
    a_gen = Model(input=[self.z,self.misc_input,self.c_input], output=a_gen_output)
    return a_gen,a_gen_output

  def createGen(self):  
    init_ = self.initializer
    dense_num = 64
    n_filters=64
    n_key_confs = self.n_key_confs

    self.dim_misc = 3
    W    = Lambda(slice_c0)(self.misc_input)
    W    = RepeatVector(self.n_key_confs)(W)
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
    a_gen = Model(input=[self.z,self.misc_input,self.c_input], output=a_gen_output)
    return a_gen,a_gen_output

  def createDisc(self):
    n_filters=64
    dense_num = 64
    init_ = self.initializer

    self.dim_misc = 3
    W    = Lambda(slice_c0)(self.misc_input)
    XW = Concatenate(axis=1)([self.a_input,W])
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

    disc_output = Dense(1, activation='linear',init=init_)(H)
    disc = Model(input=[self.a_input,self.misc_input,self.c_input],\
                  output=[disc_output],\
                  name='disc_output')
    if self.Qloss == 'adv':
      disc.compile(loss=adv_mse, optimizer=self.opt_D)
    elif self.Qloss == 'unconstrained':
      disc.compile(loss=unconstrained_mse, optimizer=self.opt_D)
    elif self.Qloss == 'hinge':
      disc.compile(loss=hinge_mse, optimizer=self.opt_D)
    return disc


    
      
