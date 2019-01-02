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
from generators.Policy import Policy,noise,G_loss
from generators.Uniform import UniformPick

from generators.slice_functions import *
from data_load_utils import load_RL_data
from sklearn.preprocessing import StandardScaler

class PlaceSup(Policy):
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
               n_score,
               explr_const,\
               tau,architecture):
    super(PlaceSup,self).__init__(sess,\
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
                                   tau,architecture)
  def createGAN(self):
    a_gen,a_gen_output=self.createGen()
    self.a_gen = a_gen
    self.a_gen.compile(loss='mse',optimizer=self.opt_G,metrics=[])
    return self.a_gen,None,None

  def createGen(self):  
    init_ = self.initializer
    dense_num = 64
    n_filters=64
    n_key_confs = self.n_key_confs

    self.dim_misc = 3 # first three, which is the current config

    W    = Lambda(slice_c0)(self.misc_input)
    W_H  = Dense(dense_num,activation='relu')(W)
    W_H  = RepeatVector(self.n_key_confs)(W_H)
    W_H  = Reshape( (self.n_key_confs,dense_num,1))(W_H)

    C_H = Reshape( (self.n_key_confs,self.dim_cvec[-1],1))(self.c_input)
    WC_H = Concatenate(axis=2)([W_H,C_H])

    H = Conv2D(filters=n_filters,\
               kernel_size=(1,4+dense_num),\
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
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H = MaxPooling2D(pool_size=(2,1))(H)
    H = Flatten()(H)
    H = Dense(dense_num,activation='relu')(H)
    H = Dense(dense_num,activation='relu')(H)
    Z_H = Dense(dense_num,activation='relu')(self.z)
    H = Concatenate()([H,Z_H])
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
    pick_pi  = UniformPick( problem.problem['env'], \
                            problem.problem['obj_region'], \
                            problem.problem['all_region'] )
 
    traj = problem.execute_policy(pick_pi,\
                                  self,\
                                  100,\
                                  key_configs=self.key_configs,\
                                  visualize=True)
 

  def train(self,place_data,n_epochs,visualize=False):
    place_states,place_actions,place_R,place_sprimes,place_sumR,place_scores\
          = self.get_and_scale_data(place_data,'place')
    c_data = place_states[0]
    w_data = place_states[1]
    
    n_data =w_data.shape[0]
    a_z = noise(n_data,self.dim_z)
    self.a_gen.fit( [a_z,w_data,c_data],\
                    place_actions,\
                    epochs=20,\
                    validation_split=0.1) 
    self.saveWeights('place_epoch')
    import pdb;pdb.set_trace()
    

  def setup_and_save_scalers(self,place_data,scaler_dir):
    place_s_cvec,place_sprime_cvec,\
    place_s_misc,place_sprime_misc,\
    place_actions,place_R,place_sumR,place_scores\
       = get_sars_data(place_data)

    self.misc_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    self.a_scaler    = StandardScaler(copy=True, with_mean=True, with_std=True)

    self.misc_scaler.fit(place_s_misc)
    self.a_scaler.fit(place_actions)
    
    print 'Saving scalers to ' + scaler_dir
    pickle.dump(self.a_scaler,open(scaler_dir+'/place_a_scaler.pkl','wb'))
    pickle.dump(self.misc_scaler,open(scaler_dir+'/place_misc_scaler.pkl','wb'))

def train_pi():
  sess = tf.Session()
  key_configs = pickle.load(open('./key_configs/key_configs.p','r'))

  place_pi = PlaceSup(sess,\
                      dim_a=3,\
                      dim_misc=9,\
                      dim_cvec=(1018,4),\
                      weight_dir='./reachability_test/sup/weights/',\
                      key_configs=key_configs,\
                      Qloss='adv',\
                      d_lr=1e-4,\
                      g_lr=1e-4,\
                      n_score=1,
                      explr_const=1000,\
                      tau=2,\
                      architecture=4)
  pick_data,place_data,traj_data = load_RL_data(n_data=5000)
  place_pi.setup_and_save_scalers(place_data = place_data,\
                                scaler_dir='./reachability_test/sup/scalers/')
  place_pi.train(place_data,n_epochs=300)

def test_pi(wpath):
  sess = tf.Session()
  key_configs = pickle.load(open('./key_configs/key_configs.p','r'))

  place_pi = PlaceSup(sess,\
                      dim_a=3,\
                      dim_misc=9,\
                      dim_cvec=(1018,4),\
                      weight_dir='./reachability_test/sup/weights/',\
                      key_configs=key_configs,\
                      Qloss='adv',\
                      d_lr=1e-4,\
                      g_lr=1e-4,\
                      n_score=1,
                      explr_const=0.1,\
                      tau=2,\
                      architecture=4)

  scaler_dir           = './reachability_test/sup/scalers/'
  place_pi.a_scaler    = pickle.load(open(scaler_dir+'/place_a_scaler.pkl','r'))
  place_pi.misc_scaler = pickle.load(open(scaler_dir+'/place_misc_scaler.pkl','r'))

  place_pi.a_gen.load_weights(wpath)
  place_pi.rollout()

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


    
      
