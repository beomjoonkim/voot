import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf 

from PickGAN import PickGAN
from sklearn.preprocessing import StandardScaler
from data_load_utils import load_pick_data

import warnings

def main():
  n_data  = int(sys.argv[1])
  d_lr = 1e-3
  g_lr = 1e-4
  n_trial = sys.argv[2]
  n_trial = n_trial+'_'+str(d_lr)+'_'+str(g_lr)

  # directory setup
  parent_dir = './pick_gan_new_data/'
  data,scalers = load_pick_data( parent_dir=parent_dir, \
                          proc_train_data_dir='processed_train_data/',\
                          n_data=n_data,n_trial=n_trial)
  x_data = data['x']; c_data = data['c']; s_data = data['s']*1

  success_idxs = s_data.squeeze()>=5


  print x_data.shape
  c0_data = data['c0']
  opose_data = data['opose'];
  oshape_data = data['oshape']
  
  w_data = np.hstack([c0_data,opose_data,oshape_data])

  dim_x   = np.shape(x_data)[1]                         # data shape
  dim_w   = np.shape(w_data)[1]                         # context vector shape
  dim_c   = (np.shape(c_data)[1],np.shape(c_data)[-1])  # collision vector shape
  
  x_scaler=scalers['x_scaler']
  session = tf.Session()
  train_results_dir =  parent_dir+'/n_data_'+str(n_data)+'/n_trial_'\
                          +str(n_trial)+'/train_results/'
  gan= PickGAN(session,dim_x,dim_w,dim_c,save_folder=train_results_dir,x_scaler=x_scaler)  

  print "Starting train"
  gan.train( x_data,\
              w_data,\
              c_data,\
              s_data,\
              epochs=300,\
              d_lr=d_lr,g_lr=g_lr)
  gan.saveWeights(additional_name='_1_')

if __name__ =='__main__':
  main()
