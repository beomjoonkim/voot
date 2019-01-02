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

def worker_p(pidx):
  command = 'python ./create_problem_instance.py'+ ' ' + str(pidx)
  
  print command
  os.system(command)

def worker_wrapper_multi_input(multi_args):
  return worker_p(multi_args)

def main():
  trials   = [int(k) for k in sys.argv[1].split(',')]
  trials = range(trials[0],trials[-1])
  n_workers = int(cpu_count())

  pool = ThreadPool(n_workers)
  results = pool.map(worker_wrapper_multi_input,trials)

if __name__ == '__main__':
  main()
