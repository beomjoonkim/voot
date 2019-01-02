import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import argparse


#from generators.PlaceDDPG import PlaceDDPG
#from generators.PlaceSAAC import PlaceSAAC
#from sup.ConstraintSup import PlaceSup
#from generators.PlaceGPS import PlaceGPS

import tensorflow as tf
from generators.Uniform import UniformPlace,UniformPick

from sklearn.preprocessing import StandardScaler
from data_load_utils       import load_RL_data,get_sars_data,get_data_dimensions
from NAMO_env import NAMO
from keras import backend as K
from openravepy import *
from train_scripts.train_algo import create_other_pi
from test_scripts.test_algo import evaluate
import socket
ROOTDIR = './'


def main():
  pick_pi  = UniformPick()
  place_pi = UniformPlace()
  problem = NAMO()
  rwd,std = evaluate( pick_pi,place_pi,visualize=False )
  print rwd,std

if __name__ == '__main__':
  main()

