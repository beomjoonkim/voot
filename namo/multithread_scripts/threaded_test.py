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
  algo     = config[0]
  n_data   = config[1]
  n_trial  = config[2]
  Qloss    = config[3]
  epoch    = config[4]
  n_score  = config[5]
  d_lr     = config[6]
  g_lr     = config[7]
  other_pi = config[8]
  explr_const = config[9]

  command = './test_with_gpu.sh ' + str(n_data) + ' ' + str(n_trial) + ' '\
            + str(algo) + ' ' + str(Qloss) + ' ' + str(n_score) +' '+ str(epoch)+ ' '\
            + str(d_lr) + ' ' + str(g_lr) + ' ' + str(other_pi) + ' ' + str(explr_const)
  print command
  os.system(command)

def worker_wrapper_multi_input(multi_args):
  return worker_p(multi_args)

def main():
  n_workers = 4 
  algo      = sys.argv[1]
  n_datas   = [int(k) for k in sys.argv[2].split(',')]
  n_datas   = range(int(n_datas[0]),int(n_datas[1])+100,100) 
  Qloss     = sys.argv[3]
  epochs    = [int(k) for k in sys.argv[4].split(',')]
  epochs    = range(int(epochs[0]),int(epochs[1])+1)
  n_score   = sys.argv[5]
  d_lr      = float(sys.argv[6])
  g_lr      = float(sys.argv[7])
  explr_const = float(sys.argv[8])
  trials    = [int(k) for k in sys.argv[9].split(',')]
  
  # Other pi???
  n_workers = cpu_count()
  configs = []
  for n_data in n_datas:
    otherpi_wfile = 'n_data_'+str(n_data)+'/onlyplace/adv/dg_lr_0.001_0.0001/n_score_5/'
    for trial in trials:
      otherpi_wfile = 'n_data_'+str(n_data)+'/onlyplace/adv/dg_lr_0.001_0.0001/n_score_5/n_trial_'+str(trial)+'/'
      for epoch in epochs:
        otherpi_wfile = 'n_data_'+str(n_data)\
          +'/onlyplace/adv/dg_lr_0.001_0.0001/n_score_5/n_trial_'\
          +str(trial)+'/train_results/'+'a_gen_epoch_'+str(epoch)+'.h5'
        configs.append([algo,n_data,trial,Qloss,epoch,n_score,d_lr,g_lr,otherpi_wfile,explr_const])
  print configs
  pool = ThreadPool(n_workers)
  results = pool.map(worker_wrapper_multi_input,configs)
  
if __name__ == '__main__':
  main()
