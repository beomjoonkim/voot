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


INFEASIBLE_SCORE = -sys.float_info.max

"""
"""

def tile(x):
  reps = [1,1,32]
  return K.tile(x,reps)

class PlaceEvaluator():
  def __init__(self,sess,dim_data,dim_context,dim_konf,save_folder):
    self.opt_G = Adam(lr=1e-4,beta_1=0.5)
    self.opt_D = Adam(lr=1e-3,beta_1=0.5) 
    self.opt_D = Adadelta()

    self.initializer = initializers.glorot_normal()
    self.sess = sess
    self.dim_data = dim_data
    self.dim_context = dim_context
    self.dim_k = dim_konf
    self.n_key_confs = dim_context[0]

    self.x_input = Input(shape=(dim_data,),name='x',dtype='float32')  
    self.w_input = Input( shape = dim_context,name='w',dtype='float32')
    self.k_input = Input( shape = dim_konf,name='k',dtype='float32')

    self.disc = self.createDisc()
    self.disc.summary()
    self.save_folder = save_folder

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
    H1=H
    H = MaxPooling2D(pool_size=(2,1))(H1)
    H = Flatten()(H)
    H = Dense(dense_num,activation='relu')(H)
    H = Dense(dense_num,activation='relu')(H)


    disc_output = Dense(1, activation='linear',init=init_)(H)
    disc = Model(input=[self.x_input,self.w_input],\
                  output=disc_output,\
                  name='disc_output')

    self.H1_model = Model(input=[self.x_input,self.w_input],\
                          output=H1,\
                          name='H1')
 
    disc.compile(loss='mse', optimizer=self.opt_D)
    return disc


