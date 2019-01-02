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

class PickDDPG(DDPGPolicy):
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
               n_score):
    super(PickDDPG,self).__init__(sess,\
                                  dim_a,\
                                  dim_misc,\
                                  dim_cvec,\
                                  weight_dir,\
                                  key_configs,\
                                  Qloss,\
                                  d_lr,\
                                  g_lr,\
                                  n_score)

  def createGen(self):
    init_ = self.initializer
    dense_num = 64
    H = Dense(dense_num,activation='relu')(self.z)
    H = Dense(dense_num,activation='relu')(H)
    H = Dense(dense_num,activation='relu')(H)
    a_gen_output = Dense(self.dim_a,
                         activation='linear',
                         init=init_,
                         name='a_gen_output')(H) 
    a_gen = Model(input=[self.z,self.misc_input,self.c_input], output=a_gen_output)
    return a_gen,a_gen_output

  def createDisc(self):
    init_ = self.initializer
    dense_num = 64
    XW = self.a_input
    H = Dense(dense_num,activation='relu')(XW)
    H = Dense(dense_num,activation='relu')(H)
    H = Dense(dense_num,activation='relu')(H)
    disc_output = Dense(1, activation='linear',init=init_)(H)
    disc = Model(input=[self.a_input,self.misc_input,self.c_input],# c_input is a dummy var
                  output=[disc_output],\
                  name='disc_output')
    disc.compile(loss='mse', optimizer=self.opt_D)
    return disc


  
