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

  command = './train_soap_with_gpu.sh '\
            + str(n_data) + ' '\
            + str(n_trial) + ' '\
            + str(gpu)

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
  trial = int(sys.argv[1])
  gpu = int(sys.argv[2])
  trials = [trial]
  gpus=[gpu]
  #n_datas=[100,200,300,400,500,600,700,900]
  n_datas = [4000,3000,2000,1000]
  n_datas = [100,500,1500,2500,3500,4500]

  configs = []
  for n_data in n_datas:
    for t in trials:
      for g in gpus:
        configs.append( ['soap',n_data,t,g] )

  print configs

  n_workers=1
  pool = ThreadPool(n_workers)
  results = pool.map(worker_wrapper_multi_input,configs)

if __name__ == '__main__':
  main()
