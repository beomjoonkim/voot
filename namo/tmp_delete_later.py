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
  n_trial = config[0]
  n_data = config[1] 

  command= 'python ./plotters/plot_planning_results.py -pi ddpg -n_data '+str(n_data) + ' -n_trial ' +str(n_trial)
  print command
  os.system(command)

def worker_wrapper_multi_input(multi_args):
  return worker_p(multi_args)

def main():
  trials    = [0,1,2,3,4,5,6,7]
  
  configs=[]
  for n_data in range(1000,11000,1000):
    for t in trials:
      configs.append([t,n_data])
  #n_workers = int(cpu_count()*2.0/3.0)
  n_workers = 1

  print 'python ./plotters/plot_planning_results.py -n_data '+str(n_data) + ' -n_trial ' +str(t)

  pool = ThreadPool(n_workers)
  results = pool.map(worker_wrapper_multi_input,configs)

if __name__ == '__main__':
  main()
