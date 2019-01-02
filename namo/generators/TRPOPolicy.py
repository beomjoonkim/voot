import matplotlib as mpl
#mpl.use('Agg') # do this before importing plt to run with no DISPLAY
import matplotlib.pyplot as plt

from keras.layers import *
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.objectives import *
from keras.callbacks import ModelCheckpoint

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

from data_load_utils import get_sars_data,reshape_data
from sklearn.preprocessing import StandardScaler
from Q_loss_functions import *
from Policy import Policy,noise

def trpo_loss(sumA_weight,old_pi_a,tau):
  def loss( actions, pi_pred ):
    p_old = K.exp(-K.sum(K.square(old_pi_a - actions),axis=-1,keepdims=True) )
    p_new = K.exp(-K.sum(K.square(pi_pred - actions),axis=-1,keepdims=True) )
    p_ratio = p_new / (p_old+1e-5)

    L_cpi     = tf.multiply(sumA_weight, p_ratio)
    clipped   = tf.clip_by_value(p_ratio,1-tau[0,0],1+tau[0,0])
    L_clipped = tf.multiply(sumA_weight,clipped)
    L         = tf.minimum(L_cpi,L_clipped) 
    return -L
  return loss

class TRPOPolicy(Policy):
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
               tau,architecture):
    super(TRPOPolicy,self).__init__(sess,dim_a,\
                                    dim_misc,\
                                    dim_cvec,\
                                    weight_dir,\
                                    key_configs,\
                                    Qloss,\
                                    d_lr,\
                                    g_lr,\
                                    n_score=None,
                                    explr_const=None,\
                                    tau=tau,architecture=architecture)


  def createGen(self):
    raise NotImplemented

  def createDisc(self):
    raise NotImplemented

  def createGAN(self): # initializes V and pi 
    disc               = self.createDisc()
    a_gen,a_gen_output = self.createGen()
    return a_gen,disc,None

  def update_pi(self,fc,misc,actions,sumA):
    # convert from list
    fc,misc,sumA = reshape_data(fc,misc,sumA)
    # reshape data 
    n_data = len(fc)

    # fit
    batch_size = np.min([32,int(n_data*0.1)])
    if batch_size == 0:
      print 'batch size too small, n_data is',n_data
      return
    print batch_size

    tau = np.tile(self.tau,(n_data,1))
    old_pi_a = self.a_gen.predict([misc,fc,sumA,actions,tau])
    checkpointer = ModelCheckpoint(filepath=self.weight_dir+'/pi_weights.hdf5',\
                                   verbose=False,\
                                   save_best_only=True,\
                                   save_weights_only=True)
    print "Fitting pi..."
    self.a_gen.fit( [misc,fc,sumA,old_pi_a,tau],\
                    actions,\
                    epochs=3,\
                    validation_split=0.1,\
                    batch_size=batch_size,callbacks=[checkpointer],\
                    verbose=False)  
    print "Done!"
    self.a_gen.load_weights(self.weight_dir+'/pi_weights.hdf5')

  def update_V(self,fc,misc,sumR):
    fc,misc,sumR = reshape_data(fc,misc,sumR)
    n_data = len(fc)
    
    batch_size = np.min([32,int(n_data*0.1)])
    if batch_size == 0:
      print 'batch size too small, n_data is',n_data
      return

    # fit
    checkpointer = ModelCheckpoint(filepath=self.weight_dir+'/weights.hdf5',\
                                   verbose=False,\
                                   save_best_only=True,\
                                   save_weights_only=True)
    print "Fitting V..."
    self.disc.fit( [misc,fc],sumR,\
                   epochs=3,\
                   validation_split=0.1,\
                   callbacks=[checkpointer],\
                   verbose=False,\
                   batch_size=batch_size) 
    print "Done!"
    self.disc.load_weights(self.weight_dir+'/weights.hdf5')

  def predict(self,cvec,misc,n_samples=1 ):
    noise_term_var = self.explr_const
    misc = self.misc_scaler.transform(misc)
    cvec = cvec.reshape((cvec.shape[0],cvec.shape[1],cvec.shape[2]))
    tau = np.tile(self.tau,(n_samples,1))

    if misc.shape[0] == 1 and n_samples > 1:
      miscs   = np.tile(misc,(n_samples,1))
      cvecs   = np.tile(cvec,(n_samples,1,1))
    else:
      miscs   = misc
      cvecs   = cvec
    dummy_sumA     = np.zeros((n_samples,1))
    dummy_old_pi_a = np.zeros((n_samples,self.dim_a))
    g = self.a_gen.predict([miscs,cvecs,dummy_sumA,dummy_old_pi_a,tau])
    noise = np.random.randn(n_samples,self.dim_a)
    g += noise_term_var*np.random.randn(n_samples,self.dim_a)
    g = self.a_scaler.inverse_transform(g)
    return g




