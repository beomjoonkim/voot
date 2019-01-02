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
  pidx    = config[3]
  tau     = config[4]


  command = 'python ./test_scripts/test_algo.py -a ' +str(algo) + ' -n_data '+ \
            str(n_data) +' -n_trial '+str(n_trial) + ' -epoch '+str(pidx) + ' -tau '+\
            str(tau)
  
  print command
  os.system(command)

def worker_wrapper_multi_input(multi_args):
  return worker_p(multi_args)

def main():
  algo_name = sys.argv[1]
  n_data   = int(sys.argv[2])
  trials   = [int(k) for k in sys.argv[3].split(',')]
  algo_dir = 'place_'+algo_name
  tau      = float(sys.argv[4])
  algo     = [algo_name]
  epochs    = range(300)
  
  configs=[]
  for t,g in zip(trials,epochs):
    configs.append([algo,n_data,t,g,tau])
  configs = itertools.product(algo,[n_data],trials,epochs,[tau])
  n_workers = int(cpu_count())

  pool = ThreadPool(n_workers)
  results = pool.map(worker_wrapper_multi_input,configs)

if __name__ == '__main__':
  main()
