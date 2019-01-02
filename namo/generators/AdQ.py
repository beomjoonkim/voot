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

from generators.PickAdQ import PickAdQ
from generators.PlaceAdQ import PlaceAdQ

from generators.Uniform import UniformPlace
from NAMO_env import NAMO
from data_load_utils import get_sars_data
from openravepy import *
from Q_loss_functions import *
from CombinedPi import CombinedPi
from data_load_utils import format_RL_data

class AdQ(CombinedPi):
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
               explr_const):
    self.eval_dir = eval_dir
    self.weight_dir = weight_dir
    self.sess =session

    print "Making place adq"
    self.place_pi = PlaceAdQ(session,\
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
                              tau=tau_place)
    print "Making pick adq"
    self.pick_pi  = PickAdQ(session,\
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
                             tau=tau_pick)

  def train(self,pick_data,place_data,plan_data,n_epochs,visualize=False):
    pick_states,pick_actions,pick_R,pick_sprimes,pick_sumR,pick_scores\
          = self.get_and_scale_data(pick_data,'pick')
    place_states,place_actions,place_R,place_sprimes,place_sumR,place_scores\
          = self.get_and_scale_data(place_data,'place')

    for i in range(n_epochs):
      print 'Completed: %.2f%%'%(i/float(n_epochs)*100)
      print 'n_data = ',len(place_states[0])

      stime = time.time()
      self.pick_pi.train_for_epoch( pick_states,pick_actions,pick_R,pick_sprimes )
      self.place_pi.train_for_epoch( place_states,place_actions,place_R,place_sprimes )
      print "Update time",time.time()-stime

      #traj_list = self.serial_rollout(visualize)
      traj_list = pickle.load(open('traj.pkl','r'))
      avg_reward  = self.record_performance( traj_list,i )
      new_pick_data  = format_RL_data( traj_list,'pick' )
      new_place_data = format_RL_data( traj_list,'place' )

      pick_states,pick_actions,pick_R,pick_sprimes,pick_sumR = self.scale_and_augment_data(pick_states,\
                                                                                 pick_actions,\
                                                                                 pick_sumR,\
                                                                                 pick_sprimes,\
                                                                                 pick_R,new_pick_data,\
                                                                                 'pick')
      place_states,place_actions,place_R,place_sprimes,place_sumR = self.scale_and_augment_data(place_states,\
                                                                                      place_actions,\
                                                                                      place_sumR,\
                                                                                      place_sprimes,\
                                                                                      place_R,\
                                                                                      new_place_data,\
                                                                                      'place')

      self.pick_pi.saveWeights('pick_epoch_'+ str(i))
      self.place_pi.saveWeights('place_epoch_'+ str(i))
      print 'Epoch time',time.time()-stime
     
  def load_weights( self,epoch ):
    print "Loading weights ",self.weight_dir+'/a_gen_pick_epoch_'+str(epoch)+'.h5'
    self.pick_pi.load_weights( self.weight_dir+'/a_gen_pick_epoch_'+str(epoch)+'.h5' )
    self.place_pi.load_weights( self.weight_dir+'/a_gen_place_epoch_'+str(epoch)+'.h5' )

    

    
