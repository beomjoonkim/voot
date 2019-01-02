import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf 

from PlaceGAN import PlaceGAN
from sklearn.preprocessing import StandardScaler
from data_load_utils import load_place_data
from train_place_soap import get_data_according_to_types,get_scalers_according_to_types,load_data
from train_utils import *

import warnings

def main():
  n_data     = get_n_data( sys.argv[1] )
  trial_name = get_trial_name( sys.argv[2] )
  parent_dir = make_parent_dir('place_gan', n_data, trial_name)

  # directory setup
  data,scalers                 = load_data(parent_dir,n_data)
  x_data,c_data,w_data,s_data  = get_data_according_to_types( data )
  dim_x,dim_w,dim_c            = get_data_dimensions(x_data,w_data,c_data)
  x_scaler                     = get_scalers_according_to_types( scalers)

  success_idxs = s_data.squeeze()>=5
  x_data = x_data[success_idxs,:]
  c_data = c_data[success_idxs,:]
  w_data = w_data[success_idxs,:]
  s_data = s_data[success_idxs,:]

  print "N_data = ",x_data.shape
  dim_x   = np.shape(x_data)[1]                         # data shape
  dim_w   = np.shape(w_data)[1]                         # context vector shape
  dim_c   = (np.shape(c_data)[1],np.shape(c_data)[2])   # collision vector shape


  session = tf.Session()
  soap = PlaceGAN(session,dim_x,dim_w,dim_c,parent_dir=parent_dir,trial_name=trial_name,
                  x_scaler=x_scaler)  

  print "Starting train"
  soap.train( x_data, \
              w_data,\
              c_data,\
              s_data, \
              epochs=300,\
              d_lr=1e-3,g_lr=1e-3)
  soap.saveWeights(additional_name='_1_')
  """
  soap.train( x_data, \
              w_data,\
              c_data,\
              s_data, \
              epochs=1000,\
              d_lr=1e-2,g_lr=1e-4 )
  soap.saveWeights(additional_name='_2_')
  """
if __name__ == '__main__':
  main()


