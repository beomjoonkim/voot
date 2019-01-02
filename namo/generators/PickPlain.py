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
from NAMO_env import NAMO
from data_load_utils import get_sars_data
from openravepy import *
from sklearn.preprocessing import StandardScaler
from Q_loss_functions import *
from Policy import Policy

def G_loss( dummy, pred ):
  return -K.mean(pred,axis=-1) # try to maximize the value of pred

def noise(n,z_size): 
  return np.random.normal(size=(n,z_size)).astype('float32')

def tile(x):
  reps = [1,1,32]
  return K.tile(x,reps)

class PickPlain(Policy):
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
               n_score,\
               explr_const,\
               tau):
    super(PickPlain,self).__init__(sess,dim_a,\
                                   dim_misc,\
                                   dim_cvec,\
                                   weight_dir,\
                                   key_configs,\
                                   Qloss,\
                                   d_lr,\
                                   g_lr,\
                                   n_score,\
                                   explr_const,\
                                   tau)

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
 
    if self.Qloss == 'adv':
      disc.compile(loss=adv_mse, optimizer=self.opt_D)
    elif self.Qloss == 'unconstrained':
      disc.compile(loss=unconstrained_mse, optimizer=self.opt_D)
    elif self.Qloss == 'hinge':
      disc.compile(loss=hinge_mse, optimizer=self.opt_D)
    return disc

  def draw_samples(self,w_data,c_data,a_data,epoch):
    a_z = noise(a_data.shape[0],self.dim_z)
    fake   = self.a_scaler.inverse_transform(self.a_gen.predict([a_z,w_data,c_data]))
    a_data = self.a_scaler.inverse_transform(a_data)

    # draw x,y
    plt.figure();
    plt.scatter(fake[:,3],fake[:,4]);
    plt.hold(True);
    plt.scatter(a_data[:,3],a_data[:,4])
    plt.savefig(self.weight_dir+'/xy_'+str(epoch)+'.png')

    # draw grasp - width and height
    plt.figure()
    plt.scatter(fake[:,1],fake[:,2]);
    plt.hold(True);
    plt.scatter(a_data[:,1],a_data[:,2])
    plt.savefig(self.weight_dir+'/g_'+str(epoch)+'.png')
    plt.close('all')

  def predict(self,cvec,misc,n_samples=1):    
    misc = self.misc_scaler.transform(misc)
    cvec = cvec.reshape((cvec.shape[0],cvec.shape[1],cvec.shape[2]))
    if misc.shape[0] == 1 and n_samples > 1:
      a_z     = noise(n_samples,self.dim_a)
      miscs   = np.tile(misc,(n_samples,1))
      cvecs   = np.tile(cvec,(n_samples,1,1))
      g       = self.a_gen.predict([a_z,miscs,cvecs])
    else:
      a_z     = noise(misc.shape[0],self.dim_a)
      g       = self.a_gen.predict([a_z,misc,cvec])
    g = self.a_scaler.inverse_transform(g)
    return g








