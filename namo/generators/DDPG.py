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

from generators.PlaceDDPG import PlaceDDPG
from generators.PickDDPG import PickDDPG

from sklearn.preprocessing import StandardScaler
from NAMO_env import NAMO
from data_load_utils import get_sars_data
from openravepy import *
from Q_loss_functions import *
from CombinedPi import CombinedPi
from data_load_utils import format_RL_data
from slice_functions import *

class DDPG(CombinedPi):
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
    self.place_pi = PlaceDDPG(session,\
                              dim_place,\
                              dim_misc,\
                              dim_cvec,\
                              weight_dir,\
                              key_configs,\
                              Qloss,\
                              d_lr_place,\
                              g_lr_place,\
                              explr_const=explr_const,\
                              tau=tau_place)
    print "Making pick ddpg"
    self.pick_pi  = PickDDPG(session,\
                             dim_pick,\
                             dim_misc,\
                             dim_cvec,\
                             weight_dir,\
                             key_configs,\
                             Qloss,\
                             d_lr_pick,\
                             g_lr_pick,\
                             explr_const=explr_const,\
                             tau=tau_place)
    self.sess=session
    self.pick_replay_buffer = {}
    self.place_replay_buffer = {}
 
  def evaluate(self,otherpi,visualize): # otherpi is a dummy var
    traj_list = []
    for n_iter in range(5):
      problem = NAMO() # different "initial" state 
      print "Executing policy..."
      traj = problem.execute_policy(self.pick_pi,\
                                    self.place_pi,\
                                    20,\
                                    key_configs=self.pick_pi.key_configs,\
                                    visualize=visualize)
      traj_list.append(traj)
      problem.env.Destroy()
      RaveDestroy()
    print 'Completed: %.2f%%'%(i/float(n_epochs)*100)
    avg_J = np.mean([np.sum(traj['r']) for traj in traj_list])
    std_J = np.std([np.sum(traj['r']) for traj in traj_list])
    pfile = open(self.eval_dir+'/test_performance_with_noise.txt','a')
    pfile.write(str(avg_J)+','+str(std_J)+'\n')
    pfile.close()


  def initialize_buffer_with_Dpl( self,pick_data,place_data ):
    pick_states,pick_actions,pick_R,pick_sprimes,pick_sumR,pick_scores,pick_aprimes\
          = self.get_and_scale_data(pick_data,'pick')
    place_states,place_actions,place_R,place_sprimes,place_sumR,place_scores,place_aprimes\
          = self.get_and_scale_data(place_data,'place')

    self.pick_replay_buffer['s'] = pick_states
    self.pick_replay_buffer['a'] = pick_actions
    self.pick_replay_buffer['r'] = pick_R
    self.pick_replay_buffer['sprime'] = pick_sprimes
    self.pick_replay_buffer['aprime'] = pick_aprimes
    
    self.place_replay_buffer['s'] = place_states
    self.place_replay_buffer['a'] = place_actions
    self.place_replay_buffer['r'] = place_R
    self.place_replay_buffer['sprime'] = place_sprimes
    self.place_replay_buffer['aprime'] = place_aprimes


  def update_buffer(self,pick_data,place_data):
    pick_states,pick_actions,pick_R,pick_sprimes,pick_sumR,pick_scores,pick_aprimes\
          = self.get_and_scale_data(pick_data,'pick')
    place_states,place_actions,place_R,place_sprimes,place_sumR,place_scores,place_aprimes\
          = self.get_and_scale_data(place_data,'place')

    #pick buffer
    self.pick_replay_buffer['s'][0] = np.vstack((self.pick_replay_buffer['s'][0],pick_states[0]))[-15000:]
    self.pick_replay_buffer['s'][1] = np.vstack((self.pick_replay_buffer['s'][1],pick_states[1]))[-15000:]
    self.pick_replay_buffer['sprime'][0] = np.vstack((self.pick_replay_buffer['sprime'][0],\
                                                      pick_sprimes[0]))[-15000:]
    self.pick_replay_buffer['sprime'][1] = np.vstack((self.pick_replay_buffer['sprime'][1],\
                                                      pick_sprimes[1]))[-15000:]
    self.pick_replay_buffer['r'] = np.vstack((self.pick_replay_buffer['r'],pick_R))[-15000:]
    self.pick_replay_buffer['aprime'].extend(pick_aprimes)
    self.pick_replay_buffer['aprime'] = self.pick_replay_buffer['aprime'][-15000:]

    #place buffer
    self.place_replay_buffer['s'][0] = np.vstack((self.place_replay_buffer['s'][0],place_states[0]))[-15000:]
    self.place_replay_buffer['s'][1] = np.vstack((self.place_replay_buffer['s'][1],place_states[1]))[-15000:]
    self.place_replay_buffer['sprime'][0] = np.vstack((self.place_replay_buffer['sprime'][0],\
                                                       place_sprimes[0]))[-15000:]
    self.place_replay_buffer['sprime'][1] = np.vstack((self.place_replay_buffer['sprime'][1],\
                                                       place_sprimes[1]))[-15000:]
    self.place_replay_buffer['r'] = np.vstack((self.place_replay_buffer['r'],place_R))[-15000:]
    self.place_replay_buffer['aprime'].extend(place_aprimes)
    self.place_replay_buffer['aprime'] = self.place_replay_buffer['aprime'][-15000:]

  def train(self,\
            pick_data,\
            place_data,\
            traj_data, # not used 
            n_epochs,
            visualize=False):

    self.initialize_buffer_with_Dpl(pick_data,place_data)

    for i in range(n_epochs):
      stime = time.time()
      # augment new data into the replay buffer
      #traj_list = self.parallel_rollout()
      traj_list = self.serial_rollout(visualize) # new traj data from the updated policy
      #traj_list = pickle.load(open('traj.pkl','r'))

      avg_reward  = self.record_performance( traj_list,i )
      new_pick_data  = format_RL_data( traj_list,'pick' )
      new_place_data = format_RL_data( traj_list,'place' )

      self.update_buffer(new_pick_data,new_place_data)

      print "Pick buffer size:",len(self.pick_replay_buffer['s'][0])
      print "Place buffer size:",len(self.place_replay_buffer['s'][0])
      print "Training place pi"
      self.place_pi.train_for_epoch( self.place_replay_buffer['s'],\
                                     self.place_replay_buffer['a'],\
                                     self.place_replay_buffer['r'],\
                                     self.place_replay_buffer['sprime'],\
                                     self.place_replay_buffer['aprime'],\
                                     otherQ = self.pick_pi.disc,\
                                     other_pi = self.pick_pi )
      print "Training pick pi"
      self.pick_pi.train_for_epoch(self.pick_replay_buffer['s'],\
                                   self.pick_replay_buffer['a'],\
                                   self.pick_replay_buffer['r'],\
                                   self.pick_replay_buffer['sprime'],\
                                   self.pick_replay_buffer['aprime'],\
                                   otherQ = self.place_pi.disc,\
                                   other_pi = self.place_pi )

      print 'Completed: %.2f%%'%(i/float(n_epochs)*100)

      self.place_pi.saveWeights('place_epoch_'+ str(i))
      self.pick_pi.saveWeights('pick_epoch_'+ str(i))
      print "Total epoch took: %.2fs"%(time.time()-stime)



