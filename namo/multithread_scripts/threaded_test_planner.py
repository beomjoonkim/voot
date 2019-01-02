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
  algo = config[0]
  pidx = config[1]
  n_trial = config[2]
  epoch   = config[3]
  n_data = config[4] 
  command = 'python ./test_scripts/test_planner.py'+ ' -pidx '+ str(pidx)+' -pi '+str(algo) \
            + ' -n_trial '+str(n_trial) + ' -epoch '+ str(epoch) + ' -n_data '+str(n_data)
  print command
  os.system(command)

def worker_wrapper_multi_input(multi_args):
  return worker_p(multi_args)

def main():
  algo  = sys.argv[1]
  n_data_to_test = int(sys.argv[2])
  #n_trial = sys.argv[3]
  #epoch  = sys.argv[4]
  
  trials    = range(1,100)
  
  configs=[]
  if algo=='soap':
    n_datas = range(1000,10000,1000)
    n_trials = [0,1,2,3]
    epochs = [[156,121,52,52],[180,99,52,156],[147,220,63,170],[49,171,24,183],\
             [87,46,144,68],[283,215,178,142],[96,269,40,26],[147,13,66,134],[61,145,264,92],\
             [86,120,181,41]]
    for ndata_idx,n_data in enumerate(n_datas):
      for ntrial_idx,n_trial in enumerate(n_trials):
        if n_data != int(n_data_to_test):
          continue
        epoch = epochs[ndata_idx][ntrial_idx]
        for t in trials:
          configs.append([algo,t,n_trial,epoch,n_data])
  elif algo == 'trpo':
   n_datas= [8000,7000,5000,4000,1000]
   if n_data_to_test==8000:
     n_trials = [0,2]
     epochs=[10,8]
   elif n_data_to_test == 7000:
     n_trials = [0,1]
     epochs = [13,6]
   elif n_data_to_test == 5000:
     n_trials = [3]
     epochs = [103]
   elif n_data_to_test == 4000:
     n_trials = [3]
     epochs=[293]
   elif n_data_to_test == 1000:
     n_trials = [1,3]
     epochs=[30,21]
   for n_data in n_datas:
     for n_trial in n_trials:
      if n_data != int(n_data_to_test):
        continue
      for epoch in epochs:
        for t in trials:
          configs.append([algo,t,n_trial,epoch,n_data])
  elif algo == 'ddpg':
    n_datas = [1000,4000,6000,8000,9000]
    if n_data_to_test==1000:
      n_trials = [2,3,6,7]
      epochs   = [107,122,96,141]
    elif n_data_to_test==4000:
      n_trials = [1,2,5,6]
      epochs = [198,174,123,182]
    elif n_data_to_test==6000:
      n_trials = [0,2,3,4]
      epochs = [156,149,220,114]
    elif n_data_to_test==8000:
      n_trials = [0]
      epochs = [106]
    elif n_data_to_test==9000:
      n_trials = [3,4,6,7]
      epochs = [114,136,244,178]
    for n_data in n_datas:
      for n_trial in n_trials:
        if n_data != int(n_data_to_test):
          continue
        for epoch in epochs:
          for t in trials:
            configs.append([algo,t,n_trial,epoch,n_data])
    

  n_workers = int(cpu_count()*2.0/3.0)

  config = configs[0]
  algo = config[0]
  pidx = config[1]
  n_trial = config[2]
  epoch   = config[3]
  n_data = config[4] 
  command = 'python ./test_scripts/test_planner.py'+ ' -pidx '+ str(pidx)+' -pi '+str(algo) \
            + ' -n_trial '+str(n_trial) + ' -epoch '+ str(epoch) + ' -n_data '+str(n_data)
  print command

  import pdb;pdb.set_trace()
  pool = ThreadPool(n_workers)
  results = pool.map(worker_wrapper_multi_input,configs)

if __name__ == '__main__':
  main()
