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
  pidx = config[0]

  command = 'python ./test_scripts/test_mcts.py -sampling_strategy voo -problem_index '+ str(pidx)
  
  print command
  os.system(command)

def worker_wrapper_multi_input(multi_args):
  return worker_p(multi_args)

def main():
  trials   = range(1000)
  
  configs=[]
  for t in trials:
    configs.append([t])
  n_workers = int(10)

  print configs
  pool = ThreadPool(n_workers)
  results = pool.map(worker_wrapper_multi_input,configs)

if __name__ == '__main__':
  main()
