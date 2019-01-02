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
import math
import shutil
import os
import random 
import pickle
import argparse

from NAMO_env import NAMO
from data_load_utils import get_sars_data
from generators.Q_loss_functions import *
from generators.Policy import Policy,noise
from generators.Uniform import UniformPlace
from generators.slice_functions import *

from data_load_utils import load_RL_data
from sklearn.preprocessing import StandardScaler


class PickSOAP(Policy):
  def __init__(self,\
               sess,
               dim_a,\
               dim_misc,\
               dim_cvec,\
               weight_dir,\
               key_configs,\
               Qloss,
               d_lr,\
               g_lr,\
               n_score,\
               explr_const,\
               tau,architecture):
  
    super(PickSOAP,self).__init__(sess,\
                                  dim_a,\
                                  dim_misc,\
                                  dim_cvec,\
                                  weight_dir,\
                                  key_configs,\
                                  Qloss,\
                                  d_lr,\
                                  g_lr,\
                                  n_score,
                                  explr_const,\
                                  tau,architecture)
  def createDisc1(self):
    pass

  def createGen1(self):
    init_       = self.initializer
    dense_num   = 64
    n_filters   = 64
    n_key_confs = self.n_key_confs
    self.dim_misc = 6

    O_xy = Lambda(slice_oxy)(self.misc_input)
    Wpose_H = Dense(dense_num,activation='relu')(O_xy)

    W_c0    = Lambda(slice_c0)(self.misc_input)
    W_c0_H  = Dense(dense_num,activation='relu')(W_c0)
  
    W_H = Concatenate(axis=1)([W_c0_H,Wpose_H])
    W_H = RepeatVector(self.n_key_confs)(W_H)
    W_H = Reshape((self.n_key_confs,dense_num*2,1))(W_H)

    C_H = Reshape( (self.n_key_confs,self.dim_cvec[-1],1))(self.c_input)
    WC_H = Concatenate(axis=2)([W_H,C_H])

    H = Conv2D(filters=n_filters,\
               kernel_size=(1,4+dense_num*2),\
               strides=(1,1),
               activation='relu')(WC_H)
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H1  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H1=H
    H = MaxPooling2D(pool_size=(2,1))(H1)
    H = Flatten()(H)
    H = Dense(dense_num,activation='relu')(H)
    H = Dense(dense_num,activation='relu')(H)
    a_gen_output = Dense(self.dim_a,
                         activation='linear',
                         init=init_,
                         name='a_gen_output')(H) 
    a_gen = Model(input=[self.z,self.misc_input,self.c_input], output=a_gen_output)
    
    return a_gen,a_gen_output

  def get_and_scale_data(self,data,data_type):
    if data is None:
      return [[],[]],[],[],[],[],[]
    s_cvec,sprime_cvec,\
    s_misc,sprime_misc,\
    actions,R,sumR,scores\
       = get_sars_data(data)

    s_misc  = self.misc_scaler.transform(s_misc)
    actions = self.a_scaler.transform(actions)
  
    states   = [s_cvec,s_misc]
    sprimes  = [sprime_cvec,sprime_misc]

    return states,actions,R,sprimes,sumR,scores

  def rollout(self):
    problem = NAMO()
    place_pi  = UniformPlace( problem.problem['env'], \
                            problem.problem['obj_region'], \
                            problem.problem['all_region'] )
 
    traj = problem.execute_policy(self,\
                                  place_pi,\
                                  100,\
                                  key_configs=self.key_configs,\
                                  visualize=True)
 
  def train(self,pick_data,n_epochs,visualize=False):
    pick_states,pick_actions,pick_sprimes,pick_R,pick_sumR,pick_scores\
          = self.get_and_scale_data(pick_data,'pick')
    print 'place n_data = ',len(pick_states[0])
    best_mse = np.inf
    for i in range(n_epochs):
      stime = time.time()
      print 'Completed: %d/%d'%(i,n_epochs) 

      _,_,mse = self.train_for_epoch( pick_states,pick_actions,pick_sumR )
      
      if mse < best_mse:
        print "Saving weights",mse,best_mse
        self.saveWeights('pick_epoch_'+str(i)+'_mse_'+str(mse))
        best_mse = mse
      print 'epoch time',time.time()-stime

  def setup_and_save_scalers(self,pick_data,scaler_dir):
    pick_s_cvec,pick_sprime_cvec,\
    pick_s_misc,pick_sprime_misc,\
    pick_actions,pick_R,pick_sumR,pick_scores\
       = get_sars_data(pick_data)

    self.misc_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    self.a_scaler    = StandardScaler(copy=True, with_mean=True, with_std=True)

    self.misc_scaler.fit(pick_s_misc)
    self.a_scaler.fit(pick_actions)
    
    print 'Saving scalers to ' + scaler_dir
    pickle.dump(self.a_scaler,open(scaler_dir+'/pick_a_scaler.pkl','wb'))
    pickle.dump(self.misc_scaler,open(scaler_dir+'/pick_misc_scaler.pkl','wb'))

def train_pi():
  sess = tf.Session()
  key_configs = pickle.load(open('./key_configs/key_configs.p','r'))

  pick_pi = PickSOAP(sess,\
                      dim_a=6,\
                      dim_misc=9,\
                      dim_cvec=(1018,4),\
                      weight_dir='./reachability_test/pick/weights/',\
                      key_configs=key_configs,\
                      Qloss='adv',\
                      d_lr=1e-4,\
                      g_lr=1e-4,\
                      n_score=1,
                      explr_const=100,\
                      tau=2,\
                      architecture=1)
  pick_data,place_data,traj_data = load_RL_data(n_data=5000)
  pick_pi.setup_and_save_scalers(pick_data = pick_data,\
                                scaler_dir='./reachability_test/pick/scalers/')
  train_err = pick_pi.train(pick_data,n_epochs=300)

def test_pi(wpath):
  sess = tf.Session()
  key_configs = pickle.load(open('./key_configs/key_configs.p','r'))

  pick_pi = PickSOAP(sess,\
                      dim_a=6,\
                      dim_misc=9,\
                      dim_cvec=(1018,4),\
                      weight_dir='./reachability_test/pick/weights/',\
                      key_configs=key_configs,\
                      Qloss='adv',\
                      d_lr=100,\
                      g_lr=100,\
                      n_score=1,
                      explr_const=0.25,\
                      tau=2,\
                      architecture=1)

  scaler_dir           = './reachability_test/pick/scalers/'
  pick_pi.a_scaler    = pickle.load(open(scaler_dir+'/pick_a_scaler.pkl','r'))
  pick_pi.misc_scaler = pickle.load(open(scaler_dir+'/pick_misc_scaler.pkl','r'))

  pick_pi.a_gen.load_weights(wpath)
  pick_pi.rollout()

def main():
  parser = argparse.ArgumentParser(description='Process configurations')
  parser.add_argument('-test',action='store_true') 
  parser.add_argument('-wpath',default='') 
  args = parser.parse_args()
  if args.test:
    test_pi(args.wpath)
  else:
    train_pi()
    

if __name__ == '__main__':
  main()


    
      
