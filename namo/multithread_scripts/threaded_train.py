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
  Qloss   = config[3]
  gpu     = config[4]
  n_score = config[5]
  d_lr    = config[6]
  g_lr    = config[7]
  tau     = config[8]
  explr_const = config[9]
  command = './train_with_gpu.sh ' + str(n_data) + ' ' + str(n_trial) + ' '\
            + str(algo) + ' ' + Qloss + ' ' +str(gpu) + ' ' + n_score + ' '\
            + str(d_lr) + ' ' + str(g_lr) + ' ' +str(tau) + ' ' + str(explr_const)
  print command
  os.system(command)

def worker_wrapper_multi_input(multi_args):
  return worker_p(multi_args)

def main():
  n_workers = 4 
  trials    = range(4)
  algo      = sys.argv[1]
  n_data    = [int(k) for k in sys.argv[2].split(',')]
  n_data    = range(n_data[0],n_data[1],100)
  Qloss     = sys.argv[3]
  gpus      = [int(k) for k in sys.argv[4].split(',')]
  n_score   = sys.argv[5]
  d_lr      = float(sys.argv[6])
  g_lr      = float(sys.argv[7])
  tau       = float(sys.argv[8])
  explr_const = float(sys.argv[9])

  configs = []
  for n in n_data:
    for t,g in zip(trials,gpus):
      configs.append([algo,n,t,Qloss,g,n_score,d_lr,g_lr,tau,explr_const])
  print configs

  pool = ThreadPool(n_workers)
  results = pool.map(worker_wrapper_multi_input,configs)


if __name__ == '__main__':
  main()
