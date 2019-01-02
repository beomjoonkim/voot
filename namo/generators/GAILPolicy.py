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
    L = tf.minimum(L_cpi,L_clipped) 
    return -L
  return loss

class GAILPolicy(Policy):
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
    super(GAILPolicy,self).__init__(sess,dim_a,\
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
    del self.disc
    self.a_gen,self.discR,self.Vfcn = self.createGAN()

  def saveWeights(self,additional_name=''):
    self.a_gen.save_weights(self.weight_dir+'/a_gen_' +additional_name+ '.h5')
    self.discR.save_weights(self.weight_dir+'/discR_' +additional_name+ '.h5')
    self.Vfcn.save_weights(self.weight_dir+'/Vfcn_' +additional_name+ '.h5')

  def createGen(self):
    raise NotImplemented

  def createDisc(self):
    raise NotImplemented

  def createGAN(self): # initializes V and pi 
    disc               = self.createDisc() #Reward function
    a_gen,a_gen_output = self.createGen()
    V = self.createV()
    return a_gen,disc,V
    
  def update_discR( self,fc_expert,misc_expert,a_expert,fc_pi,misc_pi,a_pi ):
    BATCH_SIZE = np.min([32,int(len(a_expert)*0.1),int(len(a_pi)*0.1)])
    if BATCH_SIZE==0:
      BATCH_SIZE = 1
    print BATCH_SIZE 
    
    # choose a batch of expert data
    indices      = np.random.randint(0,a_expert.shape[0],size=BATCH_SIZE)
    fc_e_batch   = np.array(fc_expert[indices,:])
    misc_e_batch = np.array(misc_expert[indices,:])
    a_e_batch    = np.array(a_expert[indices,:])

    # choose a batch of pi data
    pi_indices    = np.random.randint(0,a_pi.shape[0],size=BATCH_SIZE)
    fc_pi_batch   = np.array(fc_pi[pi_indices,:])
    misc_pi_batch = np.array(misc_pi[pi_indices,:])
    a_pi_batch    = np.array(a_pi[pi_indices,:])

    # make their scores
    fake_scores = np.zeros((BATCH_SIZE,1))
    real_scores = np.ones((BATCH_SIZE,1))

    batch_x      = np.vstack( [a_pi_batch,a_e_batch] )
    batch_fc     = np.vstack( [fc_pi_batch,fc_e_batch] )
    batch_misc   = np.vstack( [misc_pi_batch,misc_e_batch] ) 
    batch_scores = np.vstack( [fake_scores,real_scores] )

    # Update D
    print "Updating reward function..."
    self.discR.fit( {'x':batch_x,'w':batch_misc,'c':batch_fc},
                   batch_scores,
                   epochs=3, 
                   verbose=False )

    Dtrue = np.mean(self.discR.predict([a_expert,misc_expert,fc_expert]))
    Dfake = np.mean(self.discR.predict([a_pi,misc_pi,fc_pi]))
    print 'fake,real R val = (%f,%f)'%(Dfake,Dtrue)
    print "Finished an epoch"

  def update_pi(self,fc,misc,actions,sumA):
    # convert from list
    fc,misc,sumVal = reshape_data(fc,misc,sumA)
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
    """
    checkpointer = ModelCheckpoint(filepath=self.weight_dir+'/pi_weights.hdf5',\
                                   verbose=0,\
                                   save_best_only=True,\
                                   save_weights_only=True)
    """
    print "Fitting pi..."
    self.a_gen.fit( [misc,fc,sumA,old_pi_a,tau],\
                    actions,\
                    epochs=3,\
                    #validation_split=0.1,\
                    batch_size=batch_size,
                    #callbacks=[checkpointer],\
                    verbose=False)  
    print "Done!"
    #self.a_gen.load_weights(self.weight_dir+'/pi_weights.hdf5')

  def update_V(self,fc,misc,sumR):
    fc,misc,sumR = reshape_data(fc,misc,sumR)
    n_data = len(fc)
    
    batch_size = np.min([32,int(n_data*0.1)])
    if batch_size == 0:
      print 'batch size too small, n_data is',n_data
      return
    print batch_size

    # fit
    """
    checkpointer = ModelCheckpoint(filepath=self.weight_dir+'/weights.hdf5',\
                                   verbose=0,\
                                   save_best_only=True,\
                                   save_weights_only=True)
    """
    print "Fitting V..."
    self.Vfcn.fit( [misc,fc],sumR,\
                   epochs=3,\
                   verbose=False,\
    #               validation_split=0.1,\
    #               callbacks=[checkpointer],\
                   batch_size=batch_size) 
    print "Done!"
    #self.Vfcn.load_weights(self.weight_dir+'/weights.hdf5')

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




