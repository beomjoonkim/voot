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
  n_data  = config[1]
  n_trial = config[2]
  gpu     = config[3]
  tau     = config[4]
  command = 'python ./train_scripts/train_algo.py -a ' +str(algo) + ' -n_data '+ \
            str(n_data) +' -n_trial '+str(n_trial)  
  
  command = './train_with_gpu.sh ' + str(n_data) + ' ' + str(n_trial) + ' '\
            + str(algo) + ' '+ str(gpu)

  command = './train_soap_with_gpu.sh ' + str(n_data) + ' ' + str(n_trial) + ' '\
            + str(tau) + ' '+ str(gpu)

  print command +'\n'
  os.system(command)

def worker_wrapper_multi_input(multi_args):
  return worker_p(multi_args)

def main():
  n_workers = 4 
  #algo      = sys.argv[1]
  #n_datas      = [int(k) for k in sys.argv[2].split(',')]
  #gpus      = [int(k) for k in sys.argv[3].split(',')]
  #trials      = [int(k) for k in sys.argv[4].split(',')]
  k = int(sys.argv[1])
  trials = [k]
  gpus=[k]
  #n_datas=[100,200,300,400,500,600,700,900]
  n_datas = [4000,5000]
  taus = [2.0]

  configs = []
  for tau in taus:
    for n_data in n_datas:
      for t in trials:
        for g in gpus:
          configs.append( ['soap',n_data,t,g,tau] )

  print configs

  n_workers=1
  pool = ThreadPool(n_workers)
  results = pool.map(worker_wrapper_multi_input,configs)

if __name__ == '__main__':
  main()
