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
from Policy import Policy,noise,G_loss

LAMBDA=1
def augmented_mse( score_data, D_pred ):
  # Determine which of Dpred correspond to fake val  
  neg_mask      = tf.equal(score_data,INFEASIBLE_SCORE)
  y_neg         = tf.boolean_mask(D_pred,neg_mask) 
  
  # Determine which of Dpred correspond to true fcn val
  pos_mask      = tf.not_equal(score_data,INFEASIBLE_SCORE)
  y_pos         = tf.boolean_mask(D_pred,pos_mask) 
  score_pos     = tf.boolean_mask(score_data,pos_mask)

  # compute mse w.r.t true function values
  mse_on_true_data = K.mean( (K.square(score_pos - y_pos)), axis=-1)
  return mse_on_true_data+LAMBDA*K.mean( y_neg ) # try to minimize the value of y_neg

class AdQPolicy(Policy):
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
    super(AdQPolicy,self).__init__(sess,dim_a,\
                                    dim_misc,\
                                    dim_cvec,\
                                    weight_dir,\
                                    key_configs,\
                                    Qloss,\
                                    d_lr,\
                                    g_lr,\
                                    n_score,
                                    explr_const,\
                                    tau=tau)
  def createGAN(self):
    disc = self.createDisc()
    a_gen,a_gen_output = self.createGen()
    for l in disc.layers:
      l.trainable=False
    DG_output = disc([a_gen_output,self.misc_input,self.c_input]) 
    DG = Model(input=[self.misc_input,self.c_input], output=[DG_output])
    DG.compile(loss={'disc_output':G_loss,},
               optimizer=self.opt_G,
               metrics=[])
    return a_gen,disc,DG

  def train_for_epoch(self,states,actions,rewards,sprimes): 
    true_performance_list = []
    G_performance_list = []
    mse_list=[]

    c_data = states[0]
    w_data = states[1]
    c_primes = sprimes[0]
    w_primes = sprimes[1]
    a_data = actions
    r_data = rewards
    
    n_data =w_data.shape[0]
    BATCH_SIZE = np.min([32,int(len(a_data)*0.1)])
    if BATCH_SIZE==0:
      BATCH_SIZE = 1

    n_score_train = self.n_score_train
    stime=time.time()
    for idx in range(0,a_data.shape[0],BATCH_SIZE):
      BATCH_SIZE = np.min([32,int(len(actions)*0.1)])
      if BATCH_SIZE==0:
        BATCH_SIZE = 1
      #NOTE GET ADQ to work
      terminal_state_idxs = np.where(np.sum(np.sum(c_primes,axis=-1),axis=-1)==0)[0]
      nonterminal_mask = np.ones((c_primes.shape[0],1))
      nonterminal_mask[terminal_state_idxs,:] = 0

      # choose a batch of data
      indices = np.random.randint(0,a_data.shape[0],size=BATCH_SIZE)
      a_batch = np.array( a_data[indices,:] )
      w_batch = np.array( w_data[indices,:] )
      c_batch = np.array( c_data[indices,:] )
      r_batch = np.array( r_data[indices,:] )
      mask_batch   = np.array(nonterminal_mask[indices,:]) # 0 if terminal state, 1 ow
      cprime_batch = np.array( c_primes[indices,:] )
      wprime_batch = np.array( w_primes[indices,:] )

      #fake = self.a_gen.predict([sprime_batch]) # current policy's prediction
      pi_at_sprime = self.a_gen.predict([wprime_batch,cprime_batch])
      Q_at_sa_prime = self.disc.predict([pi_at_sprime,wprime_batch,cprime_batch])
      Q_targets =  r_batch+np.multiply(Q_at_sa_prime,mask_batch)

      actions_from_buffer = a_batch                              # actions executed
      # make their scores
      # minimize
      #   ||r+Q(s,\pi(s)) - Q||
      pi_action_flag = np.ones((BATCH_SIZE,1))*INFEASIBLE_SCORE # marks fake data
      a_batch_ = np.vstack( [pi_at_sprime,actions_from_buffer] )
      w_batch_ = np.vstack( [w_batch,w_batch] )
      c_batch_ = np.vstack( [c_batch,c_batch] )
      s_batch_ = np.vstack( [pi_action_flag,Q_targets] )

      self.disc.fit( {'x':a_batch_,'w':w_batch_,'c':c_batch_}, 
                     s_batch_,
                     epochs=1, 
                     verbose=True )

      # train G
      y_labels = np.ones((BATCH_SIZE,))  #dummy variable
      self.DG.fit( {'w':w_batch,'c':c_batch}, 
                   {'disc_output':y_labels,'a_gen_output':y_labels},  
                   epochs = 1, 
                   verbose=0 )  

    Dtrue = np.mean(self.disc.predict([a_data,w_data,c_data]))
    fake  = self.a_gen.predict([w_data,c_data])
    Dfake = np.mean(self.disc.predict([fake,w_data,c_data]))
    print 'fake,real disc val = (%f,%f)'%(Dfake,Dtrue)
    print 'fake,real mean val = ',np.mean(fake,axis=0),\
            np.mean(self.a_scaler.inverse_transform(a_data),axis=0)
    print "Finished an epoch"






