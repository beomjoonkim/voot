import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf 

from generators.PlaceGAN import PlaceGAN
from sklearn.preprocessing import StandardScaler
from data_load_utils import load_place_data

import warnings


def create_gan(n_data,n_trial):
  # directory setup
  data = load_place_data( parent_dir='./place_gan/', \
                          proc_train_data_dir='processed_train_data',\
                          n_data=n_data,n_trial=n_trial)
  scaled_x = data['x']
  scaled_c = data['c']
  scaled_k = data['k']
  s_data   = data['s']
  x_scaler = data['x_scaler']
  c_scaler = data['c_scaler']
  key_configs = data['key_configs']

  print "N_data = ",scaled_x.shape
  dim_data    = np.shape(scaled_x)[1]
  dim_context = (np.shape(scaled_c)[1],np.shape(scaled_c)[2]) # single channel
  dim_konf = (np.shape(scaled_k)[1],np.shape(scaled_k)[2],1)

  session = tf.Session()
  train_results_dir =  './place_soap/n_data_'+str(n_data)+'/n_trial_'\
                        +str(n_trial)+'/train_results/'
  gan = PlaceGAN(session,dim_data,dim_context,dim_konf,
                   k_data=scaled_k[0],\
                   key_configs=key_configs,\
                   x_scaler=x_scaler,\
                   c_scaler=c_scaler,\
                   save_folder=train_results_dir)  
  return gan



def main():
  n_data  = int(sys.argv[1])
  n_trial = int(sys.argv[2])

  # directory setup
  data = load_place_data( parent_dir='./place_gan/', \
                          proc_train_data_dir='processed_train_data',\
                          n_data=n_data,n_trial=0)

  scaled_x = data['x']
  scaled_c = data['c']
  scaled_k = data['k']
  s_data   = data['s']
  x_scaler = data['x_scaler']
  c_scaler = data['c_scaler']

  success_idxs = np.where(s_data.squeeze()==5)[0]
  scaled_x     = scaled_x[success_idxs,:]
  scaled_c     = scaled_c[success_idxs,:]
  s_data       = s_data[success_idxs,:]
  print "N_data = ",scaled_x.shape
  dim_data    = np.shape(scaled_x)[1]
  dim_context = (np.shape(scaled_c)[1],np.shape(scaled_c)[2]) # single channel
  dim_konf = (np.shape(scaled_k)[1],np.shape(scaled_k)[2],1)

  session = tf.Session()
  train_results_dir =  './place_gan/n_data_'+str(n_data)+'/n_trial_'\
                        +str(n_trial)+'/train_results/'
 
  print train_results_dir
  gan = PlaceGAN(session,dim_data,dim_context,dim_konf,\
                   x_scaler=x_scaler,c_scaler=c_scaler,save_folder=train_results_dir)  

  print "Starting train"
  gan.train( scaled_c,\
             scaled_k,\
             scaled_x, \
             epochs=300,\
             d_lr=1e-3,g_lr=1e-3)
  gan.saveWeights(additional_name='_1_')
  """
  gan.train( scaled_c,\
             scaled_k,\
             scaled_x, \
             epochs=1000,\
             d_lr=1e-4,g_lr=1e-4 )
  gan.saveWeights(additional_name='_2_')
  """
if __name__ == '__main__':
  main()


