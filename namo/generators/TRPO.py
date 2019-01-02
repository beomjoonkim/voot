from keras.objectives import *
from keras import backend as K
from keras import initializers
from keras.layers  import *
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

from generators.PlaceTRPO import PlaceTRPO
from generators.PickTRPO import PickTRPO

from sklearn.preprocessing import StandardScaler
from NAMO_env import NAMO
from data_load_utils import get_sars_data
from openravepy import *
from Q_loss_functions import *
from CombinedPi import CombinedPi
from data_load_utils import convert_collision_vec_to_one_hot,get_state_vals,format_RL_data
sys.path.append('../mover_library/')
from utils import clean_pose_data
import pickle

class TRPO(CombinedPi):
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
               d_lr_pick,\
               g_lr_pick,\
               d_lr_place,\
               g_lr_place,\
               tau_pick,\
               tau_place,\
               explr_const,architecture):
    self.sess = session
    self.eval_dir = eval_dir

    print "Making place trpo"
    self.place_pi = PlaceTRPO(session,\
                              dim_place,\
                              dim_misc,\
                              dim_cvec,\
                              weight_dir,\
                              key_configs,\
                              Qloss,\
                              d_lr_place,\
                              g_lr_place,\
                              tau_place,architecture=architecture)
    print "Making pick trpo"
    self.pick_pi  = PickTRPO(session,\
                             dim_pick,\
                             dim_misc,\
                             dim_cvec,\
                             weight_dir,\
                             key_configs,\
                             Qloss,\
                             d_lr_pick,\
                             g_lr_pick,\
                             tau_pick,architecture=architecture)
    self.place_pi.explr_const = explr_const
    self.pick_pi.explr_const  = explr_const
    
  def compute_Vprime(self,fc_prime,misc_prime,aprime):
    # NOTE how to make sure the data has been transformed?
    if np.sum(fc_prime) != 0:
      Vprime     = self.computeV( [misc_prime,fc_prime],aprime )
    else:
      Vprime = 0
    return Vprime

  def computeV(self,state,action):
    if action is None:
      return 0
    is_pick_state       = action.shape[-1]!=3
    if is_pick_state:
      return self.pick_pi.disc.predict(state)
    else:
      return self.place_pi.disc.predict(state)

  def update(self,pick_fc,pick_misc,pick_actions,pick_sumR,\
             place_fc,place_misc,place_actions,place_sumR,n_data,traj_data):
    self.pick_pi.update_V(  pick_fc,pick_misc,pick_sumR )      
    self.place_pi.update_V( place_fc,place_misc,place_sumR )   

    stime=time.time()
    pick_sumA,place_sumA = self.compute_sum_advantage( traj_data,n_data )
    print 'compute A time',time.time()-stime

    self.pick_pi.update_pi(  pick_fc, pick_misc,pick_actions,pick_sumA )
    self.place_pi.update_pi( place_fc,place_misc,place_actions,place_sumA)

  def train(self,\
            pick_data,\
            place_data,\
            plan_data,\
            n_epochs,\
            visualize=False):
    pick_states,pick_actions,_,_,pick_sumR,_\
          = self.get_and_scale_data(pick_data,'pick')
    place_states,place_actions,_,_,place_sumR,_\
          = self.get_and_scale_data(place_data,'place')

    pick_fc  = pick_states[0];  pick_misc  = pick_states[1]
    place_fc = place_states[0]; place_misc = place_states[1]
    plan_n_data = len(pick_fc)

    # update using the initial planning data
    stime = time.time()
    self.update( pick_fc,pick_misc,pick_actions,pick_sumR,\
                 place_fc,place_misc,place_actions,place_sumR,\
                 plan_n_data,plan_data )
    print "Update time",time.time()-stime

    for i in range(n_epochs):
      print 'Completed: %.2f%%'%(i/float(n_epochs)*100)
      stime = time.time()
      traj_list = self.serial_rollout(visualize) # new traj data from the updated policy
      #traj_list = pickle.load(open('traj.pkl','r'))
      #traj_list = self.parallel_rollout() # new traj data from the updated policy
      print "rollout time",time.time()-stime
      avg_reward = self.record_performance( traj_list,i )

      new_pick_data  = format_RL_data( traj_list,'pick' )
      new_place_data = format_RL_data( traj_list,'place' )
      pick_states,pick_actions,pick_sprimes,pick_R,pick_sumR,_ \
          = self.get_and_scale_data(new_pick_data,'pick')
      place_states,place_actions,place_sprimes,place_R,place_sumR,_ \
          = self.get_and_scale_data(new_place_data,'place')
      n_data = None

      pick_fc  = pick_states[0];  pick_misc  = pick_states[1]
      place_fc = place_states[0]; place_misc = place_states[1]

      stime = time.time()
      self.update( pick_fc,pick_misc,pick_actions,pick_sumR,\
                   place_fc,place_misc,place_actions,place_sumR,\
                   n_data,traj_list )
      self.pick_pi.saveWeights('pick_epoch_'+ str(i))
      self.place_pi.saveWeights('place_epoch_'+ str(i))
      print "Update time",time.time()-stime

