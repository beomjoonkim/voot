import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf 
import copy

from PlaceEvaluator import PlaceEvaluator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale


from preprocessing_utils import *
from data_load_utils import load_place_data
import warnings
import keras
import random

def test(x_train,w_train,c_train,s_train,\
         x_test,w_test,c_test,s_test,evaluator,train_results_dir):
  best_mse=np.inf 
  for wfile in os.listdir(train_results_dir):
    if wfile.find('.hdf5') == -1: continue
    evaluator.disc.load_weights(train_results_dir+'/'+wfile)  
    test_mse = np.mean(np.square(evaluator.disc.predict([x_test,w_test,c_test]) - s_test))
    train_mse = np.mean(np.square(evaluator.disc.predict([x_train,w_train,c_train])\
                 - s_train))
    print wfile,test_mse,train_mse
    if test_mse < best_mse:
      best_mse=test_mse
      best_train_mse = train_mse
      best_wfile = wfile
  
  print "Best values"
  print best_wfile,best_mse,best_train_mse
  return best_wfile,best_mse,best_train_mse
  sys.exit(-1)

def train(x_train,w_train,c_train,s_train,evaluator,train_results_dir): 
  stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
  checkpt = keras.callbacks.ModelCheckpoint(filepath=train_results_dir\
                                            +'/weights.{epoch:02d}.hdf5',\
                                            monitor='loss',\
                                            verbose=0, save_best_only=False, \
                                            save_weights_only=True)
  hist = evaluator.disc.fit(x=[x_train,w_train,c_train],y=s_train,\
                      batch_size=32,\
                      validation_split=0.1,\
                      epochs=200,\
                      callbacks=[checkpt,stop])

def get_train_and_test(x_data,w_data,c_data,s_data):
  n_data = len(x_data)
  n_test =int( 0.1 * n_data)
  test_idxs = random.sample(range(n_data),n_test)
  train_idxs =np.array([i for i in range(n_data) if i not in test_idxs])
  x_test = x_data[test_idxs,:]
  c_test = c_data[test_idxs,:]
  s_test = s_data[test_idxs,:]
  w_test = w_data[test_idxs,:]

  x_train = x_data[train_idxs,:]
  c_train = c_data[train_idxs,:]
  s_train = s_data[train_idxs,:]
  w_train = w_data[train_idxs,:]

  return x_train,c_train,s_train,w_train,\
         x_test,c_test,s_test,w_test

  
def train_and_test(n_data,n_trial,data_seed=None):
  proc_train_data_dir = 'processed_train_data/'
  parent_dir = './place_evaluator_new_data/n_data_'+str(n_data)
  trial_dir = parent_dir + '/n_trial_' + str(n_trial)
  scaler_dir = trial_dir + '/scalers/'
  train_results_dir = trial_dir + '/train_results/'
  if not os.path.exists(parent_dir):
    os.mkdir(parent_dir)
  if not os.path.exists(trial_dir):
    os.mkdir(trial_dir)
  if not os.path.exists(scaler_dir):
    os.mkdir(scaler_dir)
  if not os.path.exists(train_results_dir):
    os.mkdir(train_results_dir)


  data,scalers = load_place_data( parent_dir='./place_evaluator_new_data/',\
                        proc_train_data_dir='./processed_train_data',\
                        n_data=n_data,n_trial=n_trial)
  x_data = data['x']
  c_data = data['c']
  s_data = data['s']*1
  c0_data = data['c0']
  o_data = data['o'] 
  w_data = np.hstack([c0_data,o_data])

  dim_x   = np.shape(x_data)[1]                         # data shape
  dim_w   = np.shape(w_data)[1]                         # context vector shape
  dim_c   = (np.shape(c_data)[1],np.shape(c_data)[-1])   # collision vector shape
  best_mse_vals = []
  n_tests=1
  for i in range(n_tests):
    os.system( 'rm ' + train_results_dir+'/*')
    session = tf.Session()
    evaluator = PlaceEvaluator(session,dim_x,dim_w,dim_c)

    x_train,c_train,s_train,w_train,\
    x_test,c_test,s_test,w_test = get_train_and_test( x_data,w_data,c_data,s_data )


    train(x_train,w_train,c_train,s_train,evaluator,train_results_dir)
    best_wfile,test_mse,train_mse = test(x_train,w_train,c_train,s_train,\
                                x_test,w_test,c_test,s_test,evaluator,train_results_dir)
  
    best_mse_vals.append( [test_mse,train_mse] )

  best_mse_vals = np.array(best_mse_vals)
  print best_mse_vals
  print best_mse_vals.mean(axis=0)

  f = open(train_results_dir+'/best_weight.txt','w')
  f.write(best_wfile+'\n')
  f.write(str(test_mse)+'\n')
  f.write(str(train_mse)+'\n')
  f.close()

def main():
  n_data  = int(sys.argv[1])
  #n_trial = sys.argv[2]
 
  for n_trial in range(2,10):
    train_and_test(n_data,n_trial)
  #train_varying_n_data()
  #test_architecture('CNN')

if __name__ == '__main__':
  main()

