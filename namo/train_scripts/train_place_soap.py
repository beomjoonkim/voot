import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf 

from generators.PlaceSOAP import PlaceSOAP
from sklearn.preprocessing import StandardScaler
from data_load_utils import load_place_data

from train_utils import *

import warnings

def get_data_according_to_types(data):
  x_data = data['x']
  c_data = data['c'] 
  s_data = data['s']*1
  c0_data = data['c0']
  o_data = data['o'] 
  w_data = np.hstack([c0_data,o_data])
  return x_data,c_data,w_data,s_data

def get_scalers_according_to_types(scalers):
  return scalers['x_scaler']

def setup_soap( dim_x,dim_w,dim_c,parent_dir,trial_name,x_scaler):
  session = tf.Session()
  soap = PlaceSOAP(session,dim_x,dim_w,dim_c,parent_dir=parent_dir,\
                                            trial_name=trial_name,x_scaler=x_scaler)  
  return soap

def load_data( parent_dir,n_data ):
  data,scalers         = load_place_data( parent_dir=parent_dir, \
                                  proc_train_data_dir='processed_train_data',\
                                  n_data=n_data)
  return data,scalers

  
def main():
  n_data     = get_n_data( sys.argv[1] )
  trial_name = get_trial_name( sys.argv[2] )
  parent_dir = make_parent_dir( 'place_soap', n_data, trial_name )

  data,scalers                 = load_data( parent_dir,n_data )
  dim_x,dim_w,dim_c            = get_data_dimensions(x_data,w_data,c_data)
  x_data,c_data,w_data,s_data  = get_data_according_to_types( data )
  x_scaler                     = get_scalers_according_to_types( scalers)

  soap = setup_soap(dim_x,dim_w,dim_c,parent_dir,trial_name,x_scaler)
  train_soap( soap,x_data,w_data,c_data,s_data ) 


if __name__ == '__main__':
  main()


