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

from data_load_utils import get_sars_data
from sklearn.preprocessing import StandardScaler
from Q_loss_functions import *
from Policy import Policy,noise
def G_loss( dummy, pred ):
  return -K.mean(pred,axis=-1) # try to maximize the value of pred

class DDPGPolicy(Policy):
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
               explr_const,\
               tau):
    super(DDPGPolicy,self).__init__(sess,dim_a,\
                                    dim_misc,\
                                    dim_cvec,\
                                    weight_dir,\
                                    key_configs,\
                                    Qloss,\
                                    d_lr,\
                                    g_lr,\
                                    n_score=1,\
                                    explr_const=explr_const,\
                                    tau=tau,architecture=0)
    self.tau=tau

  def createGAN(self):
    if self.architecture==0:
      disc = self.createDisc()
      a_gen,a_gen_output = self.createGen()
    elif self.architecture==1:
      disc = self.createDisc1()
      a_gen,a_gen_output = self.createGen1()
    elif self.architecture==2:
      disc = self.createDisc2()
      a_gen,a_gen_output = self.createGen2()
    elif self.architecture==3:
      disc = self.createDisc3()
      a_gen,a_gen_output = self.createGen3()
    elif self.architecture==4:
      disc = self.createDisc4()
      a_gen,a_gen_output = self.createGen4()
    elif self.architecture==5:
      disc = self.createDisc5()
      a_gen,a_gen_output = self.createGen5()

    for l in disc.layers:
      l.trainable=False
    disc.summary()
    a_gen.summary()
    DG_output = disc([a_gen_output,self.misc_input,self.c_input]  )
    DG = Model(input=[self.z,self.misc_input,self.c_input], output=[DG_output])
    DG.compile(loss={'disc_output':G_loss,},
               optimizer=self.opt_G,
               metrics=[])
    return a_gen,disc,DG
  def createGen(self):
    raise NotImplemented

  def createDisc(self):
    raise NotImplemented

  def soft_update(self,network,before,after):
    new_weights = network.get_weights()
    for i in range(len(before)):
      new_weights[i] = (1-self.tau)*before[i] +\
                            (self.tau)*after[i]
    network.set_weights(new_weights)

  def compute_Q_at_sprimes(self,sprime_misc,sprime_cvec):
    is_pick_pi = self.__module__.find('Pick')

    if is_pick_pi:
      pick_pi_at_sprime = self.a_gen.predict( [sprime_misc,sprime_cvec] )
      pick_Q_at_sprime  = self.disc.predict( [pick_pi_at_sprime,sprime_misc,sprime_cvec] )
      place_pi_at_sprime = self.other_pi.a_gen.predict( [sprime_misc,sprime_cvec] )
      place_Q_at_sprime  = self.otherQ.predict( [ place_pi_at_sprime,sprime_misc,sprime_cvec])
    else:
      pick_pi_at_sprime = self.other_pi.a_gen.predict( [sprime_misc,sprime_cvec] )
      pick_Q_at_sprime  = self.otherQ.predict( [pick_pi_at_sprime,sprime_misc,sprime_cvec] )
      place_pi_at_sprime = self.a_gen.predict( [sprime_misc,sprime_cvec] )
      place_Q_at_sprime  = self.disc.predict( [ place_pi_at_sprime,sprime_misc,sprime_cvec])

    return pick_Q_at_sprime,place_Q_at_sprime

  def train_for_epoch(self,states,actions,rewards,sprimes,aprimes,otherQ,other_pi):
    self.other_pi = other_pi
    self.otherQ   = otherQ

    s_cvec = states[0]
    s_misc = states[1]

    sprime_cvec = sprimes[0]
    sprime_misc = sprimes[1]
    n_data =s_cvec.shape[0]
    BATCH_SIZE = np.min([32,int(len(actions)*0.1)])
    if BATCH_SIZE==0:
      BATCH_SIZE = 1
    print BATCH_SIZE 

    terminal_state_idxs = np.where(np.sum(sprime_misc,axis=-1)==0)[0]
    nonterminal_mask = np.ones((sprime_misc.shape[0],1))
    nonterminal_mask[terminal_state_idxs,:] = 0

    pick_aprime_mask = np.array([1 if p =='pick' else 0  for p in aprimes])
    place_aprime_mask = np.array([1 if p =='place' else 0  for p in aprimes])

    stime=time.time()
    for _ in range(50): # each rollout is 5 episodes of length 10
      # choose a batch of data - experience replay
      indices = np.random.randint(0,actions.shape[0],size=BATCH_SIZE)
      s_cvec_batch = np.array( s_cvec[indices,:] ) # collision vector
      s_misc_batch = np.array( s_misc[indices,:] )
      a_batch      = np.array( actions[indices,:] ) 
      r_batch      = np.array( rewards[indices,:] ) 
      pick_aprime_mask_batch = pick_aprime_mask[indices]
      place_aprime_mask_batch = place_aprime_mask[indices]

      pick_Q_at_sprime,place_Q_at_sprime = self.compute_Q_at_sprimes( s_misc_batch,s_cvec_batch )
      y_targets = r_batch.squeeze() \
                  + np.multiply(pick_Q_at_sprime.squeeze(),pick_aprime_mask_batch) \
                  + np.multiply(place_Q_at_sprime.squeeze(),place_aprime_mask_batch)

      # trainQ
      before = self.disc.get_weights()
      self.disc.fit( {'x':a_batch,'w':s_misc_batch,'c':s_cvec_batch},\
                      y_targets,\
                      epochs=1,\
                      verbose=False )
      after = self.disc.get_weights()
      self.soft_update(self.disc,before,after)

      # train pi
      y_labels = np.ones((BATCH_SIZE,))  # dummy variable
      before = self.a_gen.get_weights()
      self.DG.fit( {'z':a_batch,'w':s_misc_batch,'c':s_cvec_batch},  #'z' is not used for DDPG
                 {'disc_output':y_labels,'a_gen_output':y_labels},  
                 epochs = 1, 
                 verbose=0 )  
      after  = self.a_gen.get_weights()
      self.soft_update(self.a_gen,before,after)

      # Do I update the y_targets at this point? Yes, I should. But wouldn't this destabilize the 
      # training? Or is it okay because the policy is being udpated slowly? 
      # I guess the soft updates are supposed to help with destabilization. So let's try this.


    print "Epoch took: %.2fs"%(time.time()-stime)

  def predict(self,cvec,misc,n_samples=1):    
    # TODO clean pose data
    noise_term_var = self.explr_const
    misc = self.misc_scaler.transform(misc)
    cvec = cvec.reshape((cvec.shape[0],cvec.shape[1],cvec.shape[2]))
    if misc.shape[0] == 1 and n_samples > 1:
      a_z     = noise(n_samples,self.dim_a)
      miscs   = np.tile(misc,(n_samples,1))
      cvecs   = np.tile(cvec,(n_samples,1,1))
      g       = self.a_gen.predict([miscs,cvecs])
    else:
      a_z     = noise(misc.shape[0],self.dim_a)
      g       = self.a_gen.predict([misc,cvec])
    g += noise_term_var*np.random.randn(n_samples,self.dim_a)
    g = self.a_scaler.inverse_transform(g)
    return g



