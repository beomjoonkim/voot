import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf 

from generators.PickSOAP import PickSOAP
from sklearn.preprocessing import StandardScaler
from data_load_utils import load_pick_data

from train_utils import *
import warnings

def main():
  """
  n_data     = get_n_data( sys.argv[1] )
  trial_name = get_trial_name( sys.argv[2] )
  parent_dir = make_parent_dir( 'pick_soap', n_data, trial_name )


  data,scalers                 = load_data( parent_dir,n_data )
  dim_x,dim_w,dim_c            = get_data_dimensions(x_data,w_data,c_data)
  x_data,c_data,w_data,s_data  = get_data_according_to_types( data )
  x_scaler                     = get_scalers_according_to_types( scalers)
  """

  n_data  = int(sys.argv[1])
  d_lr = 1e-3
  g_lr = 1e-3
  n_trial = sys.argv[2]
  n_trial = n_trial+'_'+str(d_lr)+'_'+str(g_lr)

  # directory setup
  parent_dir = './pick_soap_new_data/'
  data,scalers = load_pick_data( parent_dir=parent_dir, \
                          proc_train_data_dir='processed_train_data/',\
                          n_data=n_data,n_trial=n_trial)
  x_data = data['x']; c_data = data['c']; s_data = data['s']*1

  c0_data = data['c0']
  opose_data = data['opose'];
  oshape_data = data['oshape']
  
  w_data = np.hstack([c0_data,opose_data,oshape_data])
  import pdb;pdb.set_trace()

  dim_x   = np.shape(x_data)[1]                         # data shape
  dim_w   = np.shape(w_data)[1]                         # context vector shape
  dim_c   = (np.shape(c_data)[1],np.shape(c_data)[-1])  # collision vector shape
  
  x_scaler=scalers['x_scaler']
  session = tf.Session()
  train_results_dir =  parent_dir+'/n_data_'+str(n_data)+'/n_trial_'\
                          +str(n_trial)+'/train_results/'
  soap = PickSOAP(session,dim_x,dim_w,dim_c,save_folder=train_results_dir,x_scaler=x_scaler)  

  print "Starting train"
  soap.train( x_data,\
              w_data,\
              c_data,\
              s_data,\
              epochs=300,\
              d_lr=d_lr,g_lr=g_lr)
  soap.saveWeights(additional_name='_1_')
if __name__ == '__main__':
  main()


