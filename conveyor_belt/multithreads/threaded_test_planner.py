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
  n_trial = config[1]

  command = 'python ./test_scripts/test_planner.py -pi ' +str(algo) +' -n_trial '+str(n_trial) 
  
  print command
  os.system(command)

def worker_wrapper_multi_input(multi_args):
  return worker_p(multi_args)

def main():
  algo_name = sys.argv[1]
  trials   = range(100)
  
  configs=[]
  for t in trials:
    configs.append([algo_name,t])
  n_workers = int(cpu_count()*1.0/3.0)

  print configs
  pool = ThreadPool(n_workers)
  results = pool.map(worker_wrapper_multi_input,configs)

if __name__ == '__main__':
  main()
