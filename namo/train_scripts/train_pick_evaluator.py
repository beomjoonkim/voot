import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf 
import copy

from PickEvaluator import PickEvaluator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale


from preprocessing_utils import *
from data_load_utils import load_pick_data
import warnings
import keras
import random
from evaluator_utils import train,test,get_train_and_test

def make_dirs(parent_dir):
  proc_train_data_dir = 'processed_train_data/'
  n_data_dir = parent_dir+'/n_data_'+str(n_data)
  trial_dir = n_data_dir + '/n_trial_' + str(n_trial)
  scaler_dir = trial_dir + '/scalers/'
  train_results_dir = trial_dir + '/train_results/'
  if not os.path.exists(parent_dir):
    os.mkdir(parent_dir)
  if not os.path.exists(n_data_dir):
    os.mkdir(n_data_dir)
  if not os.path.exists(trial_dir):
    os.mkdir(trial_dir)
  if not os.path.exists(scaler_dir):
    os.mkdir(scaler_dir)
  if not os.path.exists(train_results_dir):
    os.mkdir(train_results_dir)



def train_and_test(n_data,n_trial,parent_dir,data_seed=None):
  proc_train_data_dir = 'processed_train_data/'
  n_data_dir = parent_dir+'/n_data_'+str(n_data)
  trial_dir = n_data_dir + '/n_trial_' + str(n_trial)
  scaler_dir = trial_dir + '/scalers/'
  train_results_dir = trial_dir + '/train_results/'
  proc_train_data_dir = './processed_train_data/'
  data,scalers = load_pick_data( parent_dir=parent_dir,\
                        proc_train_data_dir=proc_train_data_dir,\
                        n_data=n_data,n_trial=n_trial)

  x_data = data['x']; c_data = data['c']; s_data = data['s']*1
  c0_data = data['c0']
  opose_data = data['opose'];
  oshape_data = data['oshape']

  w_data = np.hstack([c0_data,opose_data,oshape_data])

  print "N_data = ",x_data.shape

  dim_x   = np.shape(x_data)[1]                         # data shape
  dim_w   = np.shape(w_data)[1]                         # context vector shape
  dim_c   = (np.shape(c_data)[1],np.shape(c_data)[-1])   # collision vector shape

  session = tf.Session()
  train_results_dir = parent_dir+'/n_data_'+str(n_data)+'/n_trial_'\
                          +str(n_trial)+'/train_results/'
  best_weight_dir = train_results_dir + '/best_weights/'
  if not os.path.exists(best_weight_dir):
    os.mkdir(best_weight_dir)

  soap = PickEvaluator(session,dim_x,dim_w,dim_c)  

  best_mse_vals = []
  best_wfiles   = []
  n_tests=1
  for i in range(n_tests):
    os.system( 'rm ' + train_results_dir+'/*')
    session = tf.Session()
    evaluator = PickEvaluator(session,dim_x,dim_w,dim_c)

    x_train,c_train,s_train,w_train,\
    x_test,c_test,s_test,w_test = get_train_and_test( x_data,w_data,c_data,\
                                                            s_data )


    train(x_train,w_train,c_train,s_train,evaluator,train_results_dir)
    best_wfile,test_mse,train_mse = test(x_train,w_train,c_train,s_train,\
                                x_test,w_test,c_test,s_test,evaluator,train_results_dir)

    best_mse_vals.append( [test_mse,train_mse] )
    best_wfiles.append( best_wfile )
  print best_mse_vals
  print np.mean(np.array(best_mse_vals),axis=0)
  f = open(train_results_dir+'/best_weight.txt','w')
  f.write(best_wfile+'\n')
  f.write(str(test_mse)+'\n')
  f.write(str(train_mse)+'\n')
  f.close()

def main():
  n_data  = int(sys.argv[1])
  n_trial = sys.argv[2]
 
  train_and_test(n_data,n_trial,'./pick_evaluator_new_data/')

if __name__ == '__main__':
  main()

