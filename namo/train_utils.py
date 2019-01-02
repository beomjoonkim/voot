import numpy as np
import pickle
import os
import sys

def get_n_data( cmd_arg ):
  return int(cmd_arg)

def get_trial_name( cmd_arg ):
  return cmd_arg

def get_config_dir( parent_dir, lambda_val, n_data, d_lr,g_lr):
  return parent_dir+'/n_data_'+str(n_data)+'/'+str(lambda_val)+'_'+str(d_lr)+'_'+str(g_lr)+'/'

def make_parent_dir(algo_name, n_data, trial_name ):
  parent_dir = './'+algo_name+'_new_data'+ '/'+str(n_data) + '/'
  if not os.path.isdir(parent_dir): 
    os.mkdir( parent_dir )
  return parent_dir

def get_data_dimensions( x_data,w_data,c_data ):
  dim_x   = np.shape(x_data)[1]                         # data shape
  dim_w   = np.shape(w_data)[1]                         # context vector shape
  dim_c   = (np.shape(c_data)[1],np.shape(c_data)[2])   # collision vector shape
  return dim_x,dim_w,dim_c

def get_train_results_dir(parent_dir,lambda_val,n_data,d_lr,g_lr ):
  config_dir =  get_config_dir(parent_dir,lambda_val,n_data,d_lr,g_lr)
  train_results_dir =  config_dir +'/train_results/'
  return train_results_dir

def train_soap( soap,x_data,w_data,c_data,s_data):
  print "Starting train"
  soap.train( x_data, \
              w_data,\
              c_data,\
              s_data, \
              epochs=3000)
  soap.saveWeights(additional_name='_1_')



