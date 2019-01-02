import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf 
import argparse
import time

from generators.PlaceGPS import PlaceGPS
from sklearn.preprocessing import StandardScaler
from data_load_utils import load_place_RL_data,load_key_configs,load_guidance_data
from collect_data import search_episode,create_problem_and_env
from conveyor_belt_env import ConveyorBelt
from openravepy import *
from train_scripts.train_algo import create_soap,create_ddpg

def create_gps(n_data,n_trial):
  train_results_dir =  './place_gps/n_data_'+str(n_data)+'/n_trial_'\
                        +str(n_trial)+'/train_results/'
  key_configs = load_key_configs()
  dim_state  = (957,2,1)
  dim_action = 6
  
  session = tf.Session()
  n_traj = 5
  time_steps = 20
  policy  = PlaceGPS(session,\
                    dim_action= dim_action,\
                    dim_state = dim_state,\
                    n_plan_data = 0,\
                    n_traj=n_traj,\
                    time_steps=time_steps,\
                    save_folder=train_results_dir,\
                    key_configs=key_configs,\
                    x_scaler=None,\
                    c_scaler=None)
  return policy

def create_sup():
  pass

def evaluate_policy(policy,visualize):
  traj_list = []
  n_time_steps = 50
  problem = ConveyorBelt() # different "initial" state 
  stime = time.time()
  traj = problem.execute_policy(policy,n_time_steps,visualize)
  traj_list.append(traj)
  problem.env.Destroy()
  RaveDestroy()
  print 'Reward of a trajectory',np.sum(traj['r'])
  print 'It took %f seconds'%(time.time()-stime)

  J_list = [np.sum(traj['r']) for traj in traj_list]
  return J_list

def mk_test_results_dir(algo_dir,n_data,n_trial):
  parent_dir = algo_dir+'/n_data_'+str(n_data)
  trial_dir = parent_dir + '/n_trial_' + str(n_trial)
  test_results_dir = trial_dir +'/test_results/'
  if not os.path.exists(test_results_dir):
    os.mkdir(test_results_dir)
  return test_results_dir

def save_test_results(J_list,test_dir,test_idx):
  pickle.dump(J_list,open(test_dir+'/test_results_'+str(test_idx)+'.pkl','wb'))

def is_test_idx_already_tested(test_dir,test_idx):
  return os.path.exists(test_dir+'/test_results_'+str(test_idx)+'.pkl')

def main():
  # call this script with a parallelization
  parser = argparse.ArgumentParser(description='Process configurations')
  parser.add_argument('-n_data',type=int,default=1000)
  parser.add_argument('-n_trial',type=int,default=0)
  parser.add_argument('-a',default='gps')
  parser.add_argument('-g',action='store_true')
  parser.add_argument('-v',action='store_true')
  parser.add_argument('-test_idx',type=int,default=0) # test pidx number
  args = parser.parse_args()

  algo_dir = './place_'+args.a+'/'
  test_dir = mk_test_results_dir(algo_dir,args.n_data,args.n_trial)
  if is_test_idx_already_tested(test_dir,args.test_idx):
    print "Already tested"
    return

  if args.a == 'gps':
    policy = create_gps(args.n_data,args.n_trial)
  elif args.a == 'sup':
    policy = create_sup(args.n_data,args.n_trial)
  elif args.a == 'soap':
    policy = create_soap(args.n_data,args.n_trial)    
  elif args.a == 'ddpg':
    policy = create_ddpg(args.n_data,args.n_trial)    
    
  policy.load_weights()
  J_list = evaluate_policy( policy,args.v )
  save_test_results(J_list,test_dir,args.test_idx)
  
if __name__ == '__main__':
  main()
