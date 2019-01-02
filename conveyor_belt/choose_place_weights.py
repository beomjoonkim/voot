import sys
import os
import pickle
import time
import tensorflow as tf
import numpy as np
from PlaceSOAP import PlaceSOAP
from PlaceEvaluator import PlaceEvaluator
from openravepy import *

from keras.layers import *
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.objectives import *
from keras import backend as K
from keras import initializers

sys.path.append('../mover_library/')
from samplers import  *
#from mover_problem import sample_pick,sample_placement,pick_obj,place_obj
from operator_utils.grasp_utils import solveTwoArmIKs
from operator_utils.grasp_utils import compute_two_arm_grasp
from misc.priority_queue import Stack, Queue, FILOPriorityQueue, PriorityQueue
from TreeNode import *

from manipulation.primitives.transforms import get_point
from manipulation.primitives.savers import DynamicEnvironmentStateSaver
from manipulation.constants import FOLDED_LEFT_ARM

from data_load_utils import load_place_data
from utils import determine_best_weight_path_for_given_n_data,get_best_weight_file


def main():
  n_data = int(sys.argv[1])
  n_trial = sys.argv[2]
  sess = tf.Session()
  save_folder = './'

  data = load_place_data( parent_dir='./place_soap/',\
                          proc_train_data_dir='./processed_train_data',\
                          n_data=n_data,n_trial=n_trial)
  scaled_x = data['x']
  scaled_c = data['c']
  scaled_k = data['k']
  s_data   = data['s']
  x_scaler = data['x_scaler']

  print "N_data = ",scaled_x.shape
  dim_data    = np.shape(scaled_x)[1]
  dim_context = (np.shape(scaled_c)[1],np.shape(scaled_c)[2]) # single channel
  dim_konf = (np.shape(scaled_k)[1],np.shape(scaled_k)[2],1)

  # load SOAP
  train_results_dir = './place_soap/n_data_'+str(n_data)+'/n_trial_'\
                      +str(n_trial)+'/train_results/'
  soap = PlaceSOAP(sess,dim_data,dim_context,dim_konf,k_data=scaled_k,\
                   save_folder=train_results_dir)  
  
  # load the regressor
  path_to_best_evaluator_weight = determine_best_weight_path_for_given_n_data('./place_evaluator'\
                                                                              ,n_data)
  print "Using the evaluator weight",path_to_best_evaluator_weight
  evaluator = PlaceEvaluator(sess,dim_data,dim_context,dim_konf,train_results_dir)
  evaluator.disc.load_weights(path_to_best_evaluator_weight)
  
  # load the data
  best_score = 0
  score_list = []
  weight_fname_list = []
  weightlist = os.listdir(train_results_dir)
  if len(weightlist)==0:
    print "No trained weights in "+train_results_dir
    sys.exit(-1)
  for wfile in os.listdir(train_results_dir):
    if wfile.find('a_gen') == -1: continue
    print wfile
    soap.a_gen.load_weights( train_results_dir+wfile)
    zvals = np.random.normal(size=(n_data,soap.dim_z)).astype('float32')
    Gpred = soap.a_gen.predict( [zvals,scaled_c] )
    Epred = evaluator.disc.predict( [Gpred,scaled_c] )
    eval_score = np.mean(Epred)
    if eval_score>best_score:
      best_score=eval_score
      best_weight=wfile
      best_score=eval_score
      best_weight=wfile
      f = open(train_results_dir+'/best_weight.txt','w')
      f.write(str(best_score)+'\n')
      f.write(str(best_weight)+'\n')
      f.close()

    score_list.append( eval_score)
    weight_fname_list.append( wfile)
    print train_results_dir+wfile,eval_score
    pickle.dump([score_list,weight_fname_list], open(train_results_dir+'/weight_scores.pkl','wb'))
  print best_score,best_weight

  
if __name__ == '__main__':
  main()
