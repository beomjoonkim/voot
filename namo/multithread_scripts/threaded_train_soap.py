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
  n_data  = config[0]
  n_trial = config[1]
  gpu     =config[2]
  command = './train_soap_with_gpu.sh '\
            + str(n_data) + ' '\
            + str(n_trial) + ' '\
            + str(gpu)

  print command +'\n'
  os.system(command)

def worker_wrapper_multi_input(multi_args):
  return worker_p(multi_args)

def main():
  n_workers = 1
  #algo      = sys.argv[1]
  #n_datas      = [int(k) for k in sys.argv[2].split(',')]
  #gpus      = [int(k) for k in sys.argv[1].split(',')]
  gpus = [int(sys.argv[1])]
  #n_datas = [int(sys.argv[1])]
  #trials  = [int(k) for k in sys.argv[2].split(',')]
  trials =[int(sys.argv[1])]
  n_datas=[5000,10000]
  n_datas=[5000,10000,1000,3000,7000]

  #n_datas = [200,16000,1500,2000,2500,3000,3500,4000,4500,5000]
  configs = []
  for g in gpus:
    for n_data in n_datas:
      for t in trials:
        configs.append( [n_data,t,g] )

  print configs
  import pdb;pdb.set_trace()
  pool = ThreadPool(n_workers)
  results = pool.map(worker_wrapper_multi_input,configs)

if __name__ == '__main__':
  main()


