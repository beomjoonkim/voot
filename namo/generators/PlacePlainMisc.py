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
from Policy import Policy

class PlaceSOAP(Policy):
  def __init__(self,\
               sess,\
               dim_a,\
               dim_misc,\
               dim_cvec,\
               weight_dir,\
               perform_dir,\
               key_configs,\
               Qloss,\
               d_lr,\
               g_lr):
    super(PlaceSOAP,self).__init__(sess,\
                                   dim_a,\
                                   dim_misc,\
                                   dim_cvec,\
                                   weight_dir,\
                                   perform_dir,\
                                   key_configs,\
                                   Qloss,\
                                   d_lr,\
                                   g_lr)

  def createGen(self):  
    init_ = self.initializer
    dense_num = 64
    n_filters=64
    n_key_confs = self.n_key_confs

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
    Z_H = Dense(dense_num,activation='relu')(self.z)
    H = Concatenate()([H,Z_H])
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

    
      
