from keras.objectives import *
from keras import backend as K
from keras import initializers
from functools import partial
from keras.preprocessing.image import ImageDataGenerator
from multiprocessing.dummy import Pool as ThreadPool 

from multiprocessing import Process, Queue, Lock

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
import time

from sklearn.preprocessing import StandardScaler
from NAMO_env import NAMO
from data_load_utils import get_sars_data
from openravepy import *
from Q_loss_functions import *
from data_load_utils import format_RL_data

class CombinedPi():
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
               g_lr_place):
    raise NotImplemented

  def setup_and_save_scalers(self,pick_data,place_data,scaler_dir):
    """
    if os.path.isfile(scaler_dir+'/pick_a_scaler.pkl') and \
       os.path.isfile(scaler_dir+'/pick_misc_scaler.pkl') :
     print 'Scalers already exist'
     return
    """
    pick_s_cvec,pick_sprime_cvec,\
    pick_s_misc,pick_sprime_misc,\
    pick_actions,pick_R,pick_sumR,pick_scores,pick_aprimes\
       = get_sars_data(pick_data)

    place_s_cvec,place_sprime_cvec,\
    place_s_misc,place_sprime_misc,\
    place_actions,place_R,place_sumR,place_scores,place_aprimes\
       = get_sars_data(place_data)

    self.pick_pi.misc_scaler  = StandardScaler(copy=True, with_mean=True, with_std=True)
    self.place_pi.misc_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    self.pick_pi.a_scaler     = StandardScaler(copy=True, with_mean=True, with_std=True)
    self.place_pi.a_scaler    = StandardScaler(copy=True, with_mean=True, with_std=True)

    self.pick_pi.misc_scaler.fit(pick_s_misc)
    self.place_pi.misc_scaler.fit(place_s_misc)
    self.pick_pi.a_scaler.fit(pick_actions)
    self.place_pi.a_scaler.fit(place_actions)
    
    print 'Saving scalers to ' + scaler_dir
    pickle.dump(self.pick_pi.a_scaler,open(scaler_dir+'/pick_a_scaler.pkl','wb'))
    pickle.dump(self.place_pi.a_scaler,open(scaler_dir+'/place_a_scaler.pkl','wb'))
    pickle.dump(self.pick_pi.misc_scaler,open(scaler_dir+'/pick_misc_scaler.pkl','wb'))
    pickle.dump(self.place_pi.misc_scaler,open(scaler_dir+'/place_misc_scaler.pkl','wb'))

  def load_scalers(self,scaler_dir):
    self.pick_pi.a_scaler     = pickle.load(open(scaler_dir+'/pick_a_scaler.pkl','r'))
    self.place_pi.a_scaler    = pickle.load(open(scaler_dir+'/place_a_scaler.pkl','r'))
    self.pick_pi.misc_scaler  = pickle.load(open(scaler_dir+'/pick_misc_scaler.pkl','r'))
    self.place_pi.misc_scaler = pickle.load(open(scaler_dir+'/place_misc_scaler.pkl','r'))

  def load_weights_from_path( self,path ):
    for wfile in os.listdir( path ):
      if wfile.find( 'a_gen_pick' ) != -1:
        print "Loading pick weights",wfile
        found_pick=True
        self.pick_pi.load_weights(path+'/'+wfile)
      if wfile.find( 'a_gen_place' ) != -1 :
        print "Loading place weights",wfile
        found_place=True
        self.place_pi.load_weights(path+'/'+wfile)  

  def load_weights(self,weight_dir,epoch):
    found_pick=False
    found_place=False
    print "Trying to find weights in ",weight_dir
    for wfile in os.listdir(weight_dir):
      if wfile.find( 'pick' ) != -1 and wfile.find( str(epoch) ) !=-1:
        print "Loading pick weights",wfile
        found_pick=True
        self.pick_pi.load_weights(weight_dir+'/'+wfile)
      if wfile.find( 'place' ) != -1 and wfile.find( str(epoch) ) !=-1:
        print "Loading place weights",wfile
        found_place=True
        self.place_pi.load_weights(weight_dir+'/'+wfile)  
    return found_place*found_pick

  def get_and_scale_data(self,data,data_type):
    if data is None:
      return [[],[]],[],[],[],[],[]
    s_cvec,sprime_cvec,\
    s_misc,sprime_misc,\
    actions,R,sumR,scores,aprimes\
       = get_sars_data(data)

    #assert np.all(s_misc.mean(axis=0)  != 0)
    #assert np.all(actions.mean(axis=0) != 0)

    if data_type =='pick':
      s_misc   = self.pick_pi.misc_scaler.transform(s_misc)
      actions  = self.pick_pi.a_scaler.transform(actions)
    else:
      s_misc  = self.place_pi.misc_scaler.transform(s_misc)
      actions = self.place_pi.a_scaler.transform(actions)
  
    states   = [s_cvec,s_misc]
    sprimes  = [sprime_cvec,sprime_misc]
    #assert np.all(np.isclose(s_misc.mean(axis=0),0))
    #assert np.all(np.isclose(actions.mean(axis=0),0))

    return states,actions,R,sprimes,sumR,scores,aprimes

  def process_and_augment_data(self,states,actions,sumR,sprimes,new_data,dtype):
    nstates,nactions,nsumR,nsprimes = self.process_data( new_data,dtype )
    if len(nstates)==0:
      return states,actions,sumR,sprimes
    states[0]  = np.r_[states[0],nstates[0]]
    states[1]  = np.r_[states[1],nstates[1]]
    actions    = np.r_[actions,nactions]
    sumR       = np.r_[sumR,nsumR]
    sprimes[0] = np.r_[states[0],nsprimes[0]]
    sprimes[1] = np.r_[states[1],nsprimes[1]]
    return states,actions,sumR,sprimes

  def serial_rollout(self,visualize,n_insts=5,n_steps=10):
    traj_list = []
    # How can I pararellize this?
    for n_iter in range(n_insts):
      stime = time.time()
      problem = NAMO() # different "initial" state 
      
      print 'Problem creation time', time.time()-stime
      print "Executing policy..."
      is_pick_unif=self.pick_pi.__module__.find('Uniform') != -1
      is_place_unif=self.place_pi.__module__.find('Uniform') != -1
      if is_pick_unif:
        self.pick_pi.env          = problem.problem['env']
        self.pick_pi.robot        = problem.problem['env'].GetRobots()[0]
        self.pick_pi.obj_region   = problem.problem['obj_region']
        self.pick_pi.robot_region = problem.problem['all_region']
      if is_place_unif:
        self.place_pi.env          = problem.problem['env']
        self.place_pi.robot        = problem.problem['env'].GetRobots()[0]
        self.place_pi.obj_region   = problem.problem['obj_region']
        self.place_pi.robot_region = problem.problem['all_region']
      stime = time.time()
      traj = problem.execute_policy(self.pick_pi,\
                                    self.place_pi,\
                                    n_steps,\
                                    key_configs=self.pick_pi.key_configs,\
                                    visualize=visualize)
      print 'Episode time',time.time()-stime
      traj_list.append(traj) # add new data to traj
      problem.env.Destroy()
      RaveDestroy()
    return traj_list

  def rollout_thread(self,problem,thread_num):
    # tf tries to create a new graph for a new thread
    # So, we need to set the self.sess.graph as the default 
    # computational graph 
    # source: 
    #   https://stackoverflow.com/questions/40154320/replicating-models-in-keras-and-tensorflow-for-a-multi-threaded-setting?rq=1
    n_steps=10
    stime=time.time()
    with self.sess.graph.as_default():
      traj = problem.execute_policy(self.pick_pi,\
                                    self.place_pi,\
                                    n_steps,\
                                    key_configs=self.pick_pi.key_configs,\
                                    visualize=False)
      print 'Episode time',time.time()-stime,thread_num
    return traj

  def parallel_rollout(self):
      n_procs = 5
      pool = ThreadPool( n_procs )
      self.lock = Lock()
      procs= []
      problems=[]
      for i in range(n_procs):
        problems.append(NAMO())
      traj_list=[]
      for i in range(n_procs):
        print 'applying',i
        #traj_list.append(self.rollout_thread(problems[i],i))
        procs.append(pool.apply_async(self.rollout_thread,args=(problems[i],i,)))
      pool.close()
      pool.join()
      print [p.successful() for p in procs]
      for pidx,p in enumerate(procs):
        if not p.successful(): # Why does it ever fail? 
          print pidx,'Unsuccessful'
          traj_list.append( self.rollout_thread(problems[pidx],pidx))
        else:
          traj_list.append(p.get())

      for p in problems:
        p.env.Destroy()
        RaveDestroy()
      return traj_list

  def record_performance(self,traj_list,epoch_num):
    avg_J = np.mean([np.sum(traj['r']) for traj in traj_list])
    std_J = np.std([np.sum(traj['r']) for traj in traj_list])
    pfile = open(self.eval_dir+'/performance_with_noise.txt','a')
    pfile.write(str(epoch_num)+','+str(avg_J)+','+str(std_J)+'\n')
    pfile.close()
    return avg_J

  def get_sarsprime_from_traj_data(self,traj):
    cvecs   = traj['s_cvec']
    fvecs   = traj['f_vec']
    fcs = np.concatenate( [cvecs,fvecs], axis=-1)
    n_konf = max(np.array(fcs).shape[1:])

    self.fcs = fcs.reshape( (fcs.shape[0],n_konf,4) )
    self.miscs   = traj['s_misc']
    self.actions = traj['a']
    self.rewards = traj['r']
    eoftraj_flag = []

    self.fc_primes   = self.fcs[1:]
    self.misc_primes = self.miscs[1:]
    self.aprimes     = self.actions[1:]
    self.misc_primes.append(None)
    self.aprimes.append(None)
    self.fc_primes = np.r_[self.fc_primes,np.zeros(self.fcs[0].shape)[None,:]]

  def combine_data( traj_data,n_data ):
    fcs = []
    miscs= []
    actions = []
    rewards = []
    fc_primes = []
    misc_primes = []
    aprimes = []
    for traj in traj_data:
      self.get_sarsprime_from_traj_data(traj)
      fcs.append(self.fcs)
      miscs.append(self.miscs)
      rewards.append(self.rewards)
      actions.append(self.actions)
      fc_primes.append(self.fc_primes)
      misc_primes.append(self.misc_primes)
      aprimes.append(self.aprimes)

  def scale_and_augment_data(self,states,actions,sumR,sprimes,rewards,new_data,datatype):
    new_states,new_actions,new_rewards,new_sprimes,new_sumR,_\
          = self.get_and_scale_data(new_data,datatype)
    new_states = [np.vstack([states[0],new_states[0]]),np.vstack([states[1],new_states[1]])]
    new_actions = np.vstack([actions,new_actions])
    new_sumR = np.vstack([sumR,new_sumR])
    if rewards is None:
      new_rewards = None
    else:
      new_rewards = np.vstack([rewards,new_rewards])
    if sprimes is None:
      new_sprimes = None
    else:
      new_sprimes = [np.vstack([sprimes[0],new_sprimes[0]]),np.vstack([sprimes[1],new_sprimes[1]])]
    return new_states,new_actions,new_rewards,new_sprimes,new_sumR

  def compute_sum_advantage(self,traj_data,n_data=None):
    all_place_sumA = []
    all_pick_sumA  = []
    num_transitions = 0

    for traj in traj_data:
      self.get_sarsprime_from_traj_data(traj)
      pick_sumA  = []; place_sumA = []; adv_vals = []
      for i in range(len(self.miscs)):
        misc       = self.miscs[i]
        fc         = self.fcs[i]
        a          = self.actions[i]; 
        a          = a.reshape(1,a.shape[-1])
        r          = self.rewards[i]
        misc_prime = self.misc_primes[i]
        fc_prime   = self.fc_primes[i]
        aprime     = self.aprimes[i]

        fc   = np.array(fc)[None,:]
        misc = misc[None,:]
        fc_prime = fc_prime[None,:]
        if misc_prime is not None:
          misc_prime = misc_prime[None,:]

        is_pick_state = a.shape[-1]!=3
        is_gail = self.__module__.find("GAIL")!=-1
        if is_pick_state:
          scaled_misc = self.pick_pi.misc_scaler.transform(misc)
          if misc_prime is not None:
            scaled_misc_prime = self.place_pi.misc_scaler.transform(misc_prime)
          if is_gail:
            scaled_a      = self.pick_pi.a_scaler.transform( a )
            r = np.log(self.pick_pi.discR.predict([scaled_a,scaled_misc,fc]))
        else: 
          scaled_misc = self.place_pi.misc_scaler.transform(misc)
          if misc_prime is not None:
            scaled_misc_prime = self.place_pi.misc_scaler.transform(misc_prime)
          if is_gail:
            scaled_a = self.place_pi.a_scaler.transform( a )
            r = np.log(self.place_pi.discR.predict([scaled_a,scaled_misc,fc]))

        V      = self.computeV( [scaled_misc,fc], a) # a is needed to determine which V to use
        Vprime = self.compute_Vprime( fc_prime,scaled_misc_prime,aprime )
        advantage = r + Vprime - V
        adv_vals.append(advantage[0,0]) 
      assert( len(adv_vals)==len(self.actions) )

      for idx,r in enumerate(adv_vals):
        a    = self.actions[idx]
        sumA = np.sum(adv_vals[idx:])
        is_pick_state = a.shape[-1]!=3
        if is_pick_state:
          pick_sumA.append( sumA )
        else:
          place_sumA.append( sumA )

      all_pick_sumA.append(pick_sumA)
      all_place_sumA.append(place_sumA)
    all_pick_sumA  = np.hstack(all_pick_sumA)
    all_place_sumA = np.hstack(all_place_sumA)
    if n_data is None:
      pick_n_data = len(all_pick_sumA)
      place_n_data = len(all_place_sumA)
      return all_pick_sumA[0:pick_n_data],all_place_sumA[0:place_n_data]
    else:
      return all_pick_sumA[0:n_data],all_place_sumA[0:n_data]

