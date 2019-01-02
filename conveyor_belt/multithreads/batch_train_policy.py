import numpy as np
import scipy.io as sio
import os
import sys
import threading

from Queue import Queue
from multiprocessing.pool import ThreadPool # dummy is nothing but multiprocessing but wrapper around threading
from multiprocessing import cpu_count


import pickle
import socket
import argparse
import csv
import time
import itertools
import sys

def worker_p(config):
  algo    = config[0]
  n_data  = config[1][0]
  n_trial = config[1][1]

  if algo == 'ddpg':
    command = 'python ./train_scripts/train_algo.py -a ' +str(algo) + ' -n_data '+ \
              str(n_data) +' -n_trial '+str(n_trial)  + ' -tau 0.0001'
  else:
    command = 'python ./train_scripts/train_algo.py -a ' +str(algo) + ' -n_data '+ \
              str(n_data) +' -n_trial '+str(n_trial)  
  
  print command +'\n'
  os.system(command)

def worker_wrapper_multi_input(multi_args):
  return worker_p(multi_args)

def create_done_marker(train_results_dir):
  fin = open(train_results_dir+'/done_train.txt','a')
  fin.write( 'dummy file to mark done\n' )
  fin.close()

def determine_n_trial_and_n_data_pairs(algo_dir):
  ndir_to_train    = range(100,1000,100)
  ntrials_to_train = range(4) 
  
  to_train = []
  for ndata in ndir_to_train:
    for trial in ntrials_to_train:
      train_results = './'+algo_dir+'/n_data_'+str(ndata)+'/n_trial_'\
                      +str(trial)+'/train_results/'  
      is_dir_exists = os.path.isdir(train_results)
      is_done_training = 'done_train.txt' in os.listdir(train_results) \
                          if is_dir_exists else False
      if not is_done_training:
        to_train.append([ndata,trial])
  return to_train
      
def main():
  algo_name = sys.argv[1]
  algo_dir  = 'place_'+algo_name

  to_train = determine_n_trial_and_n_data_pairs(algo_dir)

  algo     = [algo_name]
  configs  = list(itertools.product( algo,to_train ))

  n_workers = 1
  pool = ThreadPool(n_workers)
  results = pool.map(worker_wrapper_multi_input,configs)

if __name__ == '__main__':
  main()
