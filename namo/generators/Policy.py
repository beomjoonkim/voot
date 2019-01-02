import tensorflow as tf

from keras.layers import *
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.objectives import *
from keras import backend as K
from keras import initializers

import tensorflow as tf
import sys
import numpy as np
import scipy.io as sio
import math
import csv
import shutil
import os
import random 
import pickle
import time 

from openravepy import *
from Q_loss_functions import *
from data_load_utils import get_sars_data
import matplotlib.pyplot as plt

from NAMO_env import NAMO
from sklearn.preprocessing import StandardScaler

sys.path.append('../mover_library/')
from utils import clean_pose_data
import argparse


def noise(n,z_size): 
  return np.random.normal(size=(n,z_size)).astype('float32')

def G_loss( dummy, pred ):
  return -K.mean(pred,axis=-1) # try to maximize the value of pred

class Policy(object):
  def __init__(self,sess,\
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
               tau,architecture):
    self.d_lr = d_lr
    self.g_lr = g_lr
    self.v_lr = d_lr
    self.Qloss = Qloss  

    self.architecture=architecture
    self.opt_G = Adam(lr=self.g_lr,beta_1=0.5)
    self.opt_D = Adam(lr=self.d_lr,beta_1=0.5) 
    self.opt_V = Adam(lr=self.v_lr,beta_1=0.5) # for gail

    K.set_value(self.opt_G.lr,g_lr)
    K.set_value(self.opt_D.lr,d_lr)
    
    # initialize 
    self.initializer = initializers.glorot_normal()
    self.sess = sess

    self.sess = sess
    self.dim_a = dim_a
    self.dim_misc = dim_misc
    self.dim_cvec = dim_cvec
    self.key_configs = key_configs
    self.n_key_confs = dim_cvec[0]

    # x = action 
    # w = problem instance, or context 
    # c = collision vector
    self.a_input    = Input(shape=(dim_a,),name='x',dtype='float32')  
    self.misc_input = Input(shape=(dim_misc,),name='w',dtype='float32')
    self.c_input    = Input(shape=dim_cvec,name='c',dtype='float32')

    self.tau = tau # this plays the role of epsilon
    self.tau_input = Input( shape = (1,),name='tau',dtype='float32') # collision vector

    # define inputs
    self.dim_z = dim_a
    if dim_a <10:
      dim_z = dim_a
    else:
      dim_z = int(dim_a/2)
    self.dim_z = dim_z
    self.z = Input( shape = (self.dim_z,),name='z',dtype='float32')
    self.weight_dir  = weight_dir
    self.a_gen,self.disc,self.DG = self.createGAN()

    self.n_score_train = n_score
    self.explr_const   = explr_const

  def saveWeights(self,additional_name=''):
    self.a_gen.save_weights(self.weight_dir+'/a_gen_' +additional_name+ '.h5')
    self.disc.save_weights(self.weight_dir+'/disc' +additional_name+ '.h5')

  def load_weights(self,weight_f):
    self.a_gen.load_weights(weight_f)

  def load_scalers(self,scaler_dir):
    print 'Loading scalers from '+scaler_dir
    self.a_scaler    = pickle.load(open(scaler_dir+'a_scaler.pkl','r'))
    self.misc_scaler = pickle.load(open(scaler_dir+'misc_scaler.pkl','r'))

  def predict(self,cvec,misc,n_samples=1):    
    # TODO clean pose data
    noise_term_var = self.explr_const
    misc = self.misc_scaler.transform(misc)
    cvec = cvec.reshape((cvec.shape[0],cvec.shape[1],cvec.shape[2]))
    if misc.shape[0] == 1 and n_samples > 1:
      a_z     = noise(n_samples,self.dim_a)
      miscs   = np.tile(misc,(n_samples,1))
      cvecs   = np.tile(cvec,(n_samples,1,1))
      g       = self.a_gen.predict([a_z,miscs,cvecs])
    else:
      a_z     = noise(misc.shape[0],self.dim_a)
      g       = self.a_gen.predict([a_z,misc,cvec])
    g += noise_term_var*np.random.randn(n_samples,self.dim_a)
    g = self.a_scaler.inverse_transform(g)
    return g
  
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

  def setup_and_save_scalers( self,data,scaler_dir ):
    print os.path.isfile(scaler_dir+'/a_scaler.pkl')
    print os.path.isfile(scaler_dir+'/misc_scaler.pkl')
    if os.path.isfile(scaler_dir+'/a_scaler.pkl') and \
        os.path.isfile(scaler_dir+'/misc_scaler.pkl') :
      print 'Scalers already exists'
      return
        
    s_cvec,sprime_cvec,\
    s_misc,sprime_misc,\
    actions,rewards,sumR \
       = get_sars_data(data)

    misc_scaler  = StandardScaler(copy=True, with_mean=True, with_std=True)
    a_scaler     = StandardScaler(copy=True, with_mean=True, with_std=True)

    misc_scaler.fit(s_misc)
    a_scaler.fit(actions)

    print 'Saving scalers to '+scaler_dir
    pickle.dump(a_scaler,open(scaler_dir+'/a_scaler.pkl','wb'))
    pickle.dump(misc_scaler,open(scaler_dir+'/misc_scaler.pkl','wb'))

  def evaluate( self,otherpi,visualize=False ):
    unif_otherpi  = otherpi.__module__.find('Uniform') != -1
    stime = time.time()   
    testing = False
    n_pinsts = 5
    n_time_steps = 20
    traj_list = []
    for n_iter in range(n_pinsts):
      if testing:
        prob_params = pickle.load(open('prob.pkl','r'))
        problem = NAMO(obj_poses=prob_params['obj_poses'],\
                       obj_shapes=prob_params['obj_shapes'],\
                       collided_objs=prob_params['collided_objs'],\
                       compute_orig_path=False) 
        problem.problem['original_path'] = prob_params['original_path']
      else:
        problem = NAMO()
      print "Executing P instance "+str(n_iter)+'/'+str(n_pinsts)
      if unif_otherpi:
        otherpi.init_regions( problem.problem['env'], \
                              problem.problem['obj_region'], \
                              problem.problem['all_region'] )
      if self.__module__.find('Place') != -1:
        pick_pi  = otherpi
        place_pi = self
      else:
        pick_pi  = self
        place_pi = otherpi
      traj = problem.execute_policy(pick_pi,\
                                    place_pi,n_time_steps,\
                                    key_configs=self.key_configs,\
                                    visualize=visualize)
      traj_list.append(traj)
      problem.env.Destroy()
      RaveDestroy()

    avg_J = np.mean([np.sum(traj['r']) for traj in traj_list])
    std_J = np.std([np.sum(traj['r']) for traj in traj_list])
    return avg_J,std_J

  def predict_fake_actions(self,w_data,c_data):
    n_data = len(w_data)
    a_z   = noise(n_data,self.dim_z)
    fake  = self.a_gen.predict([a_z,w_data,c_data])
    fake  = self.a_scaler.inverse_transform(fake)
    return fake

  def train_for_epoch(self,states,actions,rewards,curr_epoch=0): 
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

    if BATCH_SIZE==0:
      BATCH_SIZE = 1

    stime=time.time()
    for idx_idx,idx in enumerate(range(100)): #  I think this is more correct
      # choose a batch of data
      indices = np.random.randint(0,a_data.shape[0],size=BATCH_SIZE)
      a_batch = np.array( a_data[indices,:] )
      w_batch = np.array( w_data[indices,:] )
      c_batch = np.array( c_data[indices,:] )
      s_batch = np.array( score_data[indices,:] )

      # make predictions on w and c - call that a fake.
      # pass in Q(a,w,c) and Q(p,w,c)
      # Increase the value on the former, and decrease it on the latter.
      # We need a flag on them

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

      if curr_epoch>self.epoch_threshold: #TODO Fix this
        fake = self.predict_fake_actions( w_data,c_data )
        train_err = np.mean(np.linalg.norm(fake[idxs,:]-real_a_data[idxs,:],axis=-1))
        if train_err < self.epoch_best_mse:
          print "best weight changed",train_err
          self.epoch_best_mse = train_err
          self.epoch_best_weight = self.a_gen.get_weights()

    if curr_epoch>self.epoch_threshold:
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

  

