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
  #command = 'python mover_problem.py ' + str(config) 
  command = 'python collect_data_with_pick_soap.py 11000 0 ' +str(config)
  print command
  os.system(command)

def worker_wrapper_multi_input(multi_args):
  return worker_p(multi_args)

def main():
  train_idxs  = range(2000)
  """
  for idx in train_idxs:
    command = 'python mover_problem.py ' + str(idx) 
    os.system(command)
  """
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  n_workers = cpu_count()-3
  pool = ThreadPool(n_workers)
  print n_workers
  results = pool.map(worker_wrapper_multi_input,train_idxs)


if __name__ == '__main__':
  main()
