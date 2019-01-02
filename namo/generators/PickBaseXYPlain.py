import pickle
import math
import csv
import shutil
import os
import random 
import pandas as pd
from generators.Uniform import UniformPlace
from NAMO_env import NAMO
from data_load_utils import get_sars_data
from openravepy import *
from sklearn.preprocessing import StandardScaler
from generators.PickSOAP import *
from Policy import Policy
import sys
sys.path.append( '../mover_library/')
from utils import *
from samplers import sample_grasp_parameters,sample_ir
import copy
import time
from Policy import noise
import matplotlib.pyplot as plt

def separate_misc_data(misc_data):
  c0     = misc_data[:,0:3] # c0
  opose  = misc_data[:,3:6] 
  oshape = misc_data[:,6:9]
  return c0,opose,oshape
class PickBaseXYPlain(Policy):
  def __init__(self,sess,
              dim_a,\
              dim_misc,\
              dim_cvec,\
              weight_dir,\
              key_configs,\
              Qloss,
              d_lr,\
              g_lr,n_score):
  
    super(PickBaseXYPlain,self).__init__(sess,\
                                         dim_a,\
                                         dim_misc,\
                                         dim_cvec,\
                                         weight_dir,\
                                         key_configs,\
                                         Qloss,\
                                         d_lr,\
                                         g_lr,n_score)

  def createGen(self):
    init_ = self.initializer
    dense_num = 64
    H = Dense(dense_num,activation='relu')(self.z)
    H = Dense(dense_num,activation='relu')(H)
    H = Dense(dense_num,activation='relu')(H)
    a_gen_output = Dense(self.dim_a,
                         activation='linear',
                         init=init_,
                         name='a_gen_output')(H) 
    a_gen = Model(input=[self.z,self.misc_input,self.c_input], output=a_gen_output)
    return a_gen,a_gen_output

  def createDisc(self):
    init_ = self.initializer
    dense_num = 64
    XW = self.a_input
    H = Dense(dense_num,activation='relu')(XW)
    H = Dense(dense_num,activation='relu')(H)
    H = Dense(dense_num,activation='relu')(H)
    disc_output = Dense(1, activation='linear',init=init_)(H)
    disc = Model(input=[self.a_input,self.misc_input,self.c_input],# c_input is a dummy var
                  output=[disc_output],\
                  name='disc_output')
 
    if self.Qloss == 'adv':
      disc.compile(loss=adv_mse, optimizer=self.opt_D)
    elif self.Qloss == 'unconstrained':
      disc.compile(loss=unconstrained_mse, optimizer=self.opt_D)
    elif self.Qloss == 'hinge':
      disc.compile(loss=hinge_mse, optimizer=self.opt_D)
    return disc


  def setup_and_save_scalers(self,data,scaler_dir):
    s_cvec,sprime_cvec,\
    s_misc,sprime_misc,\
    actions,sumR\
       = get_sars_data(data)

    _,opose,_ = separate_misc_data(s_misc)
    s_misc = opose[:,:-1]
    misc_scaler  = StandardScaler(copy=True, with_mean=True, with_std=True)
    a_scaler     = StandardScaler(copy=True, with_mean=True, with_std=True)

    robot_xy = actions[:,3:-1]
    actions = robot_xy
    misc_scaler.fit(s_misc)
    a_scaler.fit(actions)

    print 'Saving scalers to '+scaler_dir
    pickle.dump(a_scaler,open(scaler_dir+'/a_scaler.pkl','wb'))
    pickle.dump(misc_scaler,open(scaler_dir+'/misc_scaler.pkl','wb'))

  def train(self,
            train_data,\
            n_epochs,
            visualize=False):
    s_cvec,sprime_cvec,\
    s_misc,sprime_misc,\
    actions,sumR\
       = get_sars_data(train_data)

    _,opose,_= separate_misc_data(s_misc)
    s_misc = opose[:,:-1]
    s_misc = self.misc_scaler.transform(s_misc)

    robot_xy = actions[:,3:-1]
    actions = robot_xy
    actions_normalized = self.a_scaler.transform(actions)  
    for i in range(n_epochs):
      stime = time.time()
      print "Training for epoch, ",i
      self.train_for_epoch(actions_normalized,\
                           s_misc,\
                           s_cvec,\
                           sumR)
      # Is the optimizer's moment changing?
      print "Epoch took: %.2fs"%(time.time()-stime)
      #self.draw_samples(s_misc,s_cvec,actions,i)
      self.saveWeights('epoch_'+ str(i))

  def draw_samples(self,w_data,c_data,a_data,epoch):
    a_z  = noise(a_data.shape[0],self.dim_z)
    fake = self.a_scaler.inverse_transform(self.a_gen.predict([a_z,w_data,c_data]))
    plt.figure();
    plt.scatter(fake[:,0],fake[:,1]);
    plt.hold(True);
    plt.scatter(a_data[:,0],a_data[:,1])
    plt.savefig(self.weight_dir+'/'+str(epoch)+'.png')
    plt.close('all')

  def predict(self,cvec,misc,n_samples=1):    
    c0     = misc[:,0:3] # c0
    opose  = misc[:,3:6] 
    oshape = misc[:,6:9]
    misc   = copy.deepcopy(opose[:,:-1])
    misc   = self.misc_scaler.transform(misc)
    cvec   = cvec.reshape((cvec.shape[0],cvec.shape[1],cvec.shape[2]))
    dangle_in_rad  = 30*np.pi/180.0   
    if misc.shape[0] == 1 and n_samples > 1:
      a_z     = noise(n_samples,self.dim_z)
      miscs   = np.tile(misc,(n_samples,1))
      cvecs   = np.tile(cvec,(n_samples,1,1))
      rel_xy  = self.a_scaler.inverse_transform(self.a_gen.predict([a_z,miscs,cvecs]))
      rel_th  = np.random.uniform(-dangle_in_rad,dangle_in_rad,n_samples)[:,None]
    else:
      a_z     = noise(misc.shape[0],self.dim_a)
      rel_xy  = self.a_scaler.inverse_transform(self.a_gen.predict([a_z,misc,cvec]))
      rel_th  = np.random.uniform(-dangle_in_rad,dangle_in_rad)

    theta,height_portion,depth_portion = sample_grasp_parameters(n_smpls=n_samples)  
    rel_robot_xyth = np.hstack([rel_xy.squeeze(),rel_th])
    if n_samples>1:
      action = np.hstack([theta[:,None],height_portion[:,None],\
                          depth_portion[:,None],rel_robot_xyth])
    else:
      action = np.hstack([theta,height_portion,depth_portion,rel_robot_xyth])[None,:]
    return action






