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
    DG_output = disc([a_gen_output,self.misc_input,self.c_input,self.tau_input]) 
    DG = Model(input=[self.z,self.misc_input,self.c_input,self.tau_input], output=[DG_output])
    DG.compile(loss={'disc_output':G_loss,},
               optimizer=self.opt_G,
               metrics=[])
    return a_gen,disc,DG

  def createDisc1(self):
    init_ = self.initializer
    dense_num = 64
    n_filters = 64

    self.dim_misc = 6

    shared_l1 = Dense(dense_num,activation='relu')
    R_xy = Lambda(slice_rxy)(self.a_input)
    O_xy = Lambda(slice_oxy)(self.misc_input)
    global_xy = Add()([R_xy,O_xy])
    R_th = Lambda(slice_rth)(self.a_input)
    Wpose_H = Concatenate(axis=1)([global_xy,R_th]) # pose feature
    Wpose_H = shared_l1(Wpose_H)

    W_c0    = Lambda(slice_c0)(self.misc_input)
    W_c0_H  = shared_l1(W_c0)
  
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

    # Grasp 
    R_grasp = Lambda(slice_grasp)(self.a_input)
    O_shape = Lambda(slice_shape)(self.misc_input)
    Wgrasp  = Concatenate(axis=1)([R_grasp,O_shape]) # pose feature

    # Combined pose and grasp
    H = Concatenate(axis=1)([H,Wgrasp])
    H = Dense(dense_num,activation='relu')(H)

    disc_output = Dense(1, activation='linear',init=init_)(H)
    disc = Model(input=[self.a_input,self.misc_input,self.c_input,self.tau_input],\
                  output=[disc_output],\
                  name='disc_output')
    disc.compile(loss=tau_loss(self.tau_input), optimizer=self.opt_D)
    return disc

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
    pick_pi  = UniformPlace( problem.problem['env'], \
                            problem.problem['obj_region'], \
                            problem.problem['all_region'] )
 
    traj = problem.execute_policy(self,\
                                  pick_pi,\
                                  100,\
                                  key_configs=self.key_configs,\
                                  visualize=True)
    return traj
 
  def train(self,pick_data,n_epochs,visualize=False):
    pick_states,pick_actions,pick_sprimes,pick_R,pick_sumR,pick_scores\
          = self.get_and_scale_data(pick_data,'pick')

    print 'pick n_data = ',len(pick_states[0])
    best_mse = np.inf
    self.best_mse_threshold = 0
    self.best_mse = np.inf
    self.epoch_threshold = 10
    for i in range(n_epochs):
      print 'Completed: %d/%d'%(i,n_epochs) 

      # Update to the epoch's best mse, if the previous epochs' and the current's differe 
      # by less than 1e-2
      stime = time.time()
      print "Tau value",self.tau
      _,_ = self.train_for_epoch( pick_states,pick_actions,pick_sumR,i )
      if i>self.epoch_threshold:
        print "Epoch update",self.best_mse,self.epoch_best_mse
        self.saveWeights('pick_epoch_'+str(i)\
                +'_'+str(self.epoch_best_mse)\
                +'_'+str(self.best_mse))

        if self.best_mse>self.epoch_best_mse+0.001: # update to prefer higher epochs
          print "Setting tau to zero,",self.best_mse-self.epoch_best_mse
          self.best_mse = self.epoch_best_mse
          self.best_mse = self.epoch_best_mse
          self.best_weight = self.epoch_best_weight
          self.tau=0
          self.best_mse_threshold = 0
        else:
          print "Setting tau to two"
          self.tau=2
          self.best_mse_threshold += 1
      print 'epoch time',time.time()-stime

      if self.best_mse_threshold == 5:
        break
     
  def train_for_epoch(self,states,actions,rewards,sprimes=None): 
    true_performance_list = []
    G_performance_list = []
    mse_list=[]
    good_reward_threshold = 2

    c_data = states[0]
    w_data = states[1]
    a_data = actions
    score_data = rewards
    
    n_data =w_data.shape[0]
    BATCH_SIZE = np.min([32,int(len(a_data)*0.1)])

    print "======New Epoch======="
    fake = self.predict_fake_actions( w_data,c_data )
    real_a_data = self.a_scaler.inverse_transform(actions)
    idxs = (rewards>good_reward_threshold).squeeze()
    self.epoch_best_mse = np.inf
    """
    problem = NAMO()
    problem.problem['env'].SetViewer('qtcoin')
    draw_configs(self.a_scaler.inverse_transform(actions)[idxs,:],problem.problem['env'],name='conf',transparency=0.5)
    import pdb;pdb.set_trace()
    """
    if BATCH_SIZE==0:
      BATCH_SIZE = 1

    n_score_train = self.n_score_train
    stime=time.time()
    #for idx_idx,idx in enumerate(range(0,a_data.shape[0],BATCH_SIZE)):
    for idx_idx,idx in enumerate(range(100)): #  I think this is more correct
      #print "%d/%d"%(idx_idx,len(range(0,a_data.shape[0],BATCH_SIZE)))
      #print 'Inter-epoch completed: %d/%d'%(idx,a_data.shape[0]) 
      # choose a batch of data
      indices = np.random.randint(0,a_data.shape[0],size=BATCH_SIZE)
      a_batch = np.array( a_data[indices,:] )
      w_batch = np.array( w_data[indices,:] )
      c_batch = np.array( c_data[indices,:] )
      s_batch = np.array( score_data[indices,:] )

      # train \hat{S}
      # make fake and reals
      a_z = noise(BATCH_SIZE,self.dim_z)
      fake = self.a_gen.predict([a_z,w_batch,c_batch])
      real = a_batch

      # make their scores
      fake_scores = np.ones((BATCH_SIZE,1))*INFEASIBLE_SCORE # marks fake data
      real_scores = s_batch
      
      a_batch_ = np.vstack( [fake,real] )
      w_batch_ = np.vstack( [w_batch,w_batch] )
      c_batch_ = np.vstack( [c_batch,c_batch] )
      s_batch_ = np.vstack( [fake_scores,real_scores] )

      tau_batch = np.tile(self.tau,(BATCH_SIZE*2,1))
      self.disc.fit( {'x':a_batch_,'w':w_batch_,'c':c_batch_,'tau':tau_batch}, 
                     s_batch_,
                     epochs=1, 
                     verbose=False )
      # train G
      a_z = noise(BATCH_SIZE,self.dim_z)
      y_labels = np.ones((BATCH_SIZE,))  #dummy variable
      tau_batch = np.tile(self.tau,(BATCH_SIZE,1))
      self.DG.fit( {'z':a_z,'w':w_batch,'c':c_batch,'tau':tau_batch}, 
                   {'disc_output':y_labels,'a_gen_output':y_labels},  
                   epochs = 1, 
                   verbose=0 )  

      if sprimes>self.epoch_threshold: #TODO Fix this
        fake = self.predict_fake_actions( w_data,c_data )
        train_err = np.mean(np.linalg.norm(fake[idxs,:]-real_a_data[idxs,:],axis=-1))
        if train_err < self.epoch_best_mse:
          print "best weight changed",train_err
          self.epoch_best_mse = train_err
          self.epoch_best_weight = self.a_gen.get_weights()

    if sprimes>self.epoch_threshold:
      self.a_gen.set_weights(self.epoch_best_weight)

    a_z  = noise(n_data,self.dim_z)
    fake = self.a_gen.predict([a_z,w_data,c_data])

    # Q values
    tau_batch = np.tile(self.tau,(n_data,1))
    Dtrue = np.mean(self.disc.predict([a_data,w_data,c_data,tau_batch]))
    Dfake = np.mean(self.disc.predict([fake,w_data,c_data,tau_batch]))
    print 'fake,real disc val = (%f,%f)'%(Dfake,Dtrue)

    # Real values
    fake  = self.a_scaler.inverse_transform(fake)
    print 'fake,real mean val = ',np.mean(fake,axis=0),np.mean(real_a_data,axis=0)
    print 'fake,real std val = ',np.std(fake,axis=0),np.std(real,axis=0)
    
    #print 'Preproc max val = ',\
    #  np.sort(np.max(self.preproc.predict([w_data,c_data]).squeeze(),axis=0))[-10:]
    print "Finished an epoch"
  
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

def train_pi(weight_dir,n_data):
  sess = tf.Session()
  key_configs = pickle.load(open('./key_configs/key_configs.p','r'))

  pick_pi = PickSOAP(sess,\
                      dim_a=6,\
                      dim_misc=9,\
                      dim_cvec=(1018,4),\
                      weight_dir=weight_dir,\
                      key_configs=key_configs,\
                      Qloss='adv',\
                      d_lr=1e-4,\
                      g_lr=1e-4,\
                      n_score=1,
                      explr_const=100,\
                      tau=2,\
                      architecture=1)
  pick_data,place_data,traj_data = load_RL_data(n_data=n_data)

  scaler_dir = weight_dir + '/scalers/'
  os.makedirs(scaler_dir)
  pick_pi.setup_and_save_scalers(pick_data = pick_data,\
                                 scaler_dir=scaler_dir)
  pick_pi.train(pick_data,n_epochs=300)

def test_pi(weight_dir,wpath):
  sess = tf.Session()
  key_configs = pickle.load(open('./key_configs/key_configs.p','r'))
  pick_pi = PickSOAP(sess,\
                      dim_a=6,\
                      dim_misc=9,\
                      dim_cvec=(1018,4),\
                      weight_dir=weight_dir,\
                      key_configs=key_configs,\
                      Qloss='adv',\
                      d_lr=100,\
                      g_lr=100,\
                      n_score=1,
                      explr_const=0.5,\
                      tau=2,\
                      architecture=1)

  scaler_dir          = weight_dir + '/scalers/'
  pick_pi.a_scaler    = pickle.load(open(scaler_dir+'/pick_a_scaler.pkl','r'))
  pick_pi.misc_scaler = pickle.load(open(scaler_dir+'/pick_misc_scaler.pkl','r'))
  pick_pi.a_gen.load_weights(weight_dir+wpath)
  pick_pi.rollout()

def main():
  parser = argparse.ArgumentParser(description='Process configurations')
  parser.add_argument('-test',action='store_true') 
  parser.add_argument('-wpath',default='') 
  parser.add_argument('-n_trial',type=int,default=0) 
  parser.add_argument('-n_data',type=int,default=100) 
  args = parser.parse_args()


  weight_dir='./reachability_test/pick/weights/n_data_'\
                +str(args.n_data)\
                +'/n_trial_'+str(args.n_trial)+'/'

  if args.test:
    test_pi(weight_dir,args.wpath)
  else:
    if os.path.isdir(weight_dir): 
      shutil.rmtree(weight_dir)
    train_pi(weight_dir,args.n_data)
    

if __name__ == '__main__':
  main()


    
      
