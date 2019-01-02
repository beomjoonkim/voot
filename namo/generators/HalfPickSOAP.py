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

from generators.Uniform import UniformPlace
from NAMO_env import NAMO
from data_load_utils import get_sars_data
from openravepy import *
from Q_loss_functions import *
from CombinedPi import CombinedPi


class HalfPickSOAP(CombinedPi):
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

    print "Making place soap"
    problem = NAMO()
    self.place_pi = UniformPlace( problem.problem['env'], \
                                  problem.problem['obj_region'], \
                                  problem.problem['all_region'] )

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
                             tau=tau_pick,architecture=architecture)

  def train(self,pick_data,place_data,plan_data,n_epochs,visualize=False):
    pick_states,pick_actions,pick_sprimes,pick_R,pick_sumR,pick_scores,_\
          = self.get_and_scale_data(pick_data,'pick')

    best_mse = np.inf
    self.best_mse_threshold = 0
    self.best_mse = np.inf
    self.pick_pi.epoch_threshold = 10
    for i in range(n_epochs):
      print 'Completed: %d/%d'%(i,n_epochs) 
      stime = time.time()
      print "Tau value",self.pick_pi.tau
      self.pick_pi.train_for_epoch( pick_states,pick_actions,pick_sumR,i )
      if i>self.pick_pi.epoch_threshold:
        print "Epoch update",self.best_mse,self.pick_pi.epoch_best_mse
        self.pick_pi.saveWeights('pick_epoch_'+str(i)\
                +'_'+str(self.pick_pi.epoch_best_mse)\
                +'_'+str(self.best_mse))

        if self.best_mse>self.pick_pi.epoch_best_mse+0.001: # update to prefer higher epochs
          print "Setting tau to zero,",self.best_mse-self.pick_pi.epoch_best_mse
          self.best_mse = self.pick_pi.epoch_best_mse
          self.best_weight = self.pick_pi.epoch_best_weight
          #self.pick_pi.tau=0
          self.best_mse_threshold = 0
        else:
          print "Setting tau to two"
          #self.pick_pi.tau=2
          self.best_mse_threshold += 1
      print 'epoch time',time.time()-stime

      #if self.best_mse_threshold == 5:
      #  break

    """
    for i in range(n_epochs):
      stime = time.time()
      print 'Completed: %d/%d'%(i,n_epochs) #(/float(n_epochs)*100)

      Dfake,Dtrue,_=self.pick_pi.train_for_epoch( pick_states,pick_actions,pick_sumR )
      #if i == 10:
      #  self.serial_rollout(visualize,n_steps=100)
      self.pick_pi.saveWeights('pick_epoch_'+ str(i))
      if Dfake==Dtrue:
        pfile = open(self.eval_dir+'/SameD.txt','a')
        pfile.write(str(i)+','+str(Dfake)+','+str(Dtrue)+'\n')
        pfile.close()
 
      print 'epoch time',time.time()-stime
    """
     
  def load_weights( self,epoch ):
    for wfile in os.listdir(self.weight_dir):
      if wfile.find( 'pick' ) != -1 and wfile.find( 'epoch_'+str(epoch) ) !=-1\
        and wfile.find('a_gen')!=-1:
        break
    print "Loading pick weights",self.weight_dir+wfile
    self.pick_pi.a_gen.load_weights( self.weight_dir+wfile )

    

    
