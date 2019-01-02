from keras.layers import *
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.objectives import *
from keras import backend as K
from keras import initializers
from functools import partial
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import sys

import numpy as np
import scipy.io as sio
import time
import pickle
import math
import csv
import shutil
import os
import random 
from sklearn.preprocessing import StandardScaler

from generators.PickSOAP import PickSOAP
from generators.PlaceSOAP import PlaceSOAP
from generators.PickPlain import PickPlain

from NAMO_env import NAMO
from data_load_utils import get_sars_data
from openravepy import *
from Q_loss_functions import *
from CombinedPi import CombinedPi

class SOAP(CombinedPi):
  def __init__(self,\
               session,\
               dim_pick,\
               dim_place,\
               dim_cvec,\
               dim_misc,\
               weight_dir,\
               eval_dir,\
               key_configs,\
               Qloss,\
               tau_pick,\
               tau_place,\
               d_lr_pick,\
               g_lr_pick,\
               d_lr_place,\
               g_lr_place,
               explr_const,architecture):
    self.eval_dir = eval_dir
    self.weight_dir = weight_dir
    self.sess =session
    #assert architecture==0,'SOAP uses a fixed architecture for pick and place'
    #assert architecture==0,'SOAP uses a fixed architecture for pick and place'
    

    print "Making place soap"
    self.place_pi = PlaceSOAP(session,\
                              dim_place,\
                              dim_misc,\
                              dim_cvec,\
                              weight_dir,\
                              key_configs,\
                              Qloss,\
                              d_lr_place,\
                              g_lr_place,\
                              n_score=1,\
                              explr_const=explr_const,\
                              tau=tau_place,architecture=4)

    if architecture == 0:
      print "Making pick soap"
      self.pick_pi  = PickSOAP(session,\
                               dim_pick,\
                               dim_misc,\
                               dim_cvec,\
                               weight_dir,\
                               key_configs,\
                               Qloss,\
                               d_lr_pick,\
                               g_lr_pick,\
                               n_score=1,\
                               explr_const=explr_const,\
                               tau=tau_pick,architecture=1)
    elif architecture ==1:
      print "Making pick soap"
      self.pick_pi  = PickSOAP(session,\
                               dim_pick,\
                               dim_misc,\
                               dim_cvec,\
                               weight_dir,\
                               key_configs,\
                               Qloss,\
                               d_lr_pick,\
                               g_lr_pick,\
                               n_score=1,\
                               explr_const=explr_const,\
                               tau=tau_pick,architecture=0)

  def train(self,pick_data,place_data,plan_data,n_epochs,visualize=False):
    pick_states,pick_actions,pick_sprimes,pick_R,pick_sumR,pick_scores\
          = self.get_and_scale_data(pick_data,'pick')
    place_states,place_actions,place_sprimes,place_R,place_sumR,place_scores\
          = self.get_and_scale_data(place_data,'place')
    print 'pick n_data = ',len(place_states[0])
    print 'place n_data = ',len(place_states[0])
    for i in range(n_epochs):
      stime = time.time()
      print 'Completed: %d/%d'%(i,n_epochs) #(/float(n_epochs)*100)

      _,_,place_mse = self.place_pi.train_for_epoch( place_states,place_actions,place_sumR )
      _,_,pick_mse  = self.pick_pi.train_for_epoch( pick_states,pick_actions,pick_sumR )
      print place_mse,pick_mse
      import pdb;pdb.set_trace()

      #if i==10:
      #  self.serial_rollout(visualize)

      self.pick_pi.saveWeights('pick_epoch_'+ str(i))
      self.place_pi.saveWeights('place_epoch_'+ str(i))
      print 'epoch time',time.time()-stime
     
  def load_weights( self,epoch ):
    print "Loading weights ",self.weight_dir+'/a_gen_pick_epoch_'+str(epoch)+'.h5'
    self.pick_pi.load_weights( self.weight_dir+'/a_gen_pick_epoch_'+str(epoch)+'.h5' )
    print "Loading weights ",self.weight_dir+'/a_gen_place_epoch_'+str(epoch)+'.h5'
    self.place_pi.load_weights( self.weight_dir+'/a_gen_place_epoch_'+str(epoch)+'.h5' )

    

    
