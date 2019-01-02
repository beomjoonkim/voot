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

  

if __name__ == '__main__':
  main()

