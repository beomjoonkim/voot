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

from generators.PlaceGAIL import PlaceGAIL
from generators.PickGAIL import PickGAIL

from sklearn.preprocessing import StandardScaler
from NAMO_env import NAMO
from data_load_utils import get_sars_data
from openravepy import *
from Q_loss_functions import *
from CombinedPi import CombinedPi
from data_load_utils import convert_collision_vec_to_one_hot,get_state_vals,format_RL_data
sys.path.append('../mover_library/')
from utils import clean_pose_data

class GAIL(CombinedPi):
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
               explr_const,\
               architecture):
    self.eval_dir = eval_dir
    self.sess = session
    print "Making place gail"
    self.place_pi = PlaceGAIL(session,\
                              dim_place,\
                              dim_misc,\
                              dim_cvec,\
                              weight_dir,\
                              key_configs,\
                              Qloss,\
                              d_lr_place,\
                              g_lr_place,\
                              tau_place,architecture)
    print "Making pick gail"
    self.pick_pi  = PickGAIL(session,\
                             dim_pick,\
                             dim_misc,\
                             dim_cvec,\
                             weight_dir,\
                             key_configs,\
                             Qloss,\
                             d_lr_pick,\
                             g_lr_pick,\
                             tau_pick,architecture)

    self.place_pi.explr_const = explr_const
    self.pick_pi.explr_const  = explr_const
    
  def computeV(self,state,action):
    is_pick_state       = action.shape[-1]!=3
    if is_pick_state:
      return self.pick_pi.Vfcn.predict(state)
    else:
      return self.place_pi.Vfcn.predict(state)

  def compute_Vprime(self,fc_prime,misc_prime,aprime):                                                             
    if np.sum(fc_prime) != 0:
      Vprime     = self.computeV( [misc_prime,fc_prime],aprime )
    else:
      Vprime = 0
    return Vprime

  def compute_r_and_sumR(self,trajs,action_type,n_data):
    all_sum_rewards=[]
    all_rewards=[]
    num_transitions = 0

    for i,tau in enumerate(trajs):
      step_idxs = range(len(tau['s_cvec']))
      if len(tau['a'])==0: continue
      rewards = []; sum_rewards = []
      traj_step_idxs   = []
      for s_cvec,f_vec,s_misc,a,r,step_idx in zip(tau['s_cvec'],\
                                                  tau['f_vec'],\
                                                  tau['s_misc'],\
                                                  tau['a'],\
                                                  tau['r'],\
                                                  step_idxs):
        is_pick_action = a.shape[-1]==6
        if is_pick_action:
          if action_type != 'pick':
            continue
          misc_scaler = self.pick_pi.misc_scaler
          a_scaler    = self.pick_pi.a_scaler
          pi          = self.pick_pi
        else:
          if action_type != 'place':
            continue
          misc_scaler = self.place_pi.misc_scaler
          a_scaler    =self.place_pi.a_scaler
          pi = self.place_pi
        # create fc vec
        s_cvec = s_cvec.reshape(( s_cvec.shape[0],s_cvec.shape[1],s_cvec.shape[2] ))
        f_vec  = f_vec.reshape(( f_vec.shape[0],f_vec.shape[1],f_vec.shape[2] ))
        s_fc = np.concatenate([s_cvec,f_vec],axis=-1 )
        s_misc  = s_misc.reshape((1,np.max(s_misc.shape))) 
        a       = a.reshape((1,np.max(a.shape)))
        s_misc_transformed = misc_scaler.transform(s_misc)
        a_transformed      = a_scaler.transform(a)
        learned_r = np.log(pi.discR.predict([a_transformed,s_misc_transformed,s_fc]))
       
        # add them to the trajectory 
        rewards.append( learned_r[0,0] )

      sum_rewards.append( [np.sum(rewards[idx:]) for idx,r in enumerate(rewards)] )
      all_sum_rewards.append( np.array([sum_rewards]).squeeze()) 
      all_rewards.append( np.array(rewards).squeeze() )
      num_transitions += len(sum_rewards) 

    all_sum_rewards  = np.hstack(all_sum_rewards)
    all_rewards      = np.hstack(all_rewards)

    if n_data is None:
      n_data = len(all_rewards)

    return all_rewards[:n_data],all_sum_rewards[:n_data]

  def train(self,\
            planner_pick_data,\
            planner_place_data,\
            plan_data,\
            n_epochs,\
            visualize=False):

    planner_pick_states,planner_pick_actions,planner_pick_sprimes,\
    planner_pick_R,planner_pick_sumR,_\
          = self.get_and_scale_data(planner_pick_data,'pick')

    planner_place_states,planner_place_actions,\
    planner_place_sprimes,planner_place_R,planner_place_sumR,_\
          = self.get_and_scale_data(planner_place_data,'place')

    planner_pick_fc  = planner_pick_states[0];  planner_pick_misc  = planner_pick_states[1]
    planner_place_fc = planner_place_states[0]; planner_place_misc = planner_place_states[1]
    plan_n_data = len(planner_pick_fc)
    self.pick_pi.saveWeights('pick_epoch_'+ str(0))
    self.place_pi.saveWeights('place_epoch_'+ str(0))

    for i in range(n_epochs):
      print 'Completed: %.2f%%'%(i/float(n_epochs)*100)
      stime = time.time()
      #traj_list = self.parallel_rollout() # new traj data from the updated policy
      traj_list = self.serial_rollout(visualize) # new traj data from the updated policy
      #traj_list = pickle.load(open('traj.pkl','r'))
      print "rollout time",time.time()-stime
      avg_reward = self.record_performance( traj_list,i )

      print "Using rollout data"
      new_pick_data  = format_RL_data( traj_list,'pick' )
      new_place_data = format_RL_data( traj_list,'place' )

      pick_states,pick_actions,pick_R,pick_sprimes,_,_ \
          = self.get_and_scale_data(new_pick_data,'pick')
      place_states,place_actions,place_R,place_sprimes,_,_ \
          = self.get_and_scale_data(new_place_data,'place')
      n_data = None

      pick_fc  = pick_states[0];  pick_misc  = pick_states[1]
      place_fc = place_states[0]; place_misc = place_states[1]

      is_no_pick_data = new_pick_data is None
      is_no_place_data = new_place_data is None

      # Update discR - the learned reward function
      if not is_no_pick_data:
        print "Updating pick reward function"
        self.pick_pi.update_discR( planner_pick_fc,planner_pick_misc,planner_pick_actions,\
                                 pick_fc,pick_misc,pick_actions )
      if not is_no_place_data:
        print "Updating place reward function"
        self.place_pi.update_discR( planner_place_fc,planner_place_misc,planner_place_actions,\
                                  place_fc,place_misc,place_actions )
      # Replace the rewards with the learned one
      pi_pick_r,pi_pick_sumR   = self.compute_r_and_sumR(traj_list,'pick',n_data)  
      pi_place_r,pi_place_sumR = self.compute_r_and_sumR(traj_list,'place',n_data)

      # Update V
      if not is_no_pick_data:
        print "Updating pick V function"
        self.pick_pi.update_V(  pick_fc,pick_misc,pi_pick_sumR )      
      if not is_no_place_data:
        print "Updating place V function"
        self.place_pi.update_V( place_fc,place_misc,pi_place_sumR )

      pick_sumA,place_sumA = self.compute_sum_advantage( traj_list,n_data  )

      # Update pi
      if not is_no_pick_data:
        self.pick_pi.update_pi(  pick_fc,pick_misc,pick_actions,pick_sumA )
      if not is_no_place_data:
        self.place_pi.update_pi( place_fc,place_misc,place_actions,place_sumA )

      self.pick_pi.saveWeights('pick_epoch_'+ str(i))
      self.place_pi.saveWeights('place_epoch_'+ str(i))
      print "Update complete"


