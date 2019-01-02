import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
import socket

from generators.PlaceGPS import PlaceGPS
from generators.PlaceSOAP import PlaceSOAP
from generators.PlaceDDPG import PlaceDDPG
from generators.PlaceAdQ import PlaceAdQ
from generators.PlaceTRPO import PlaceTRPO
from generators.PlaceUniform import PlaceUnif
from generators.PlaceGAIL import PlaceGAIL

from sklearn.preprocessing import StandardScaler
from data_load_utils import load_place_RL_data,load_key_configs, \
                            load_guidance_data,load_place_data,\
                            format_RL_data,setup_save_dirs

from collect_data import search_episode,create_problem_and_env
from conveyor_belt_env import ConveyorBelt
from openravepy import *
#from multithreads.threaded_train_policy import determine_n_trial_and_n_data_pairs

ROOTDIR = './'

if socket.gethostname() == 'dell-XPS-15-9560':
  ROOTDIR      = '../../AdvActorCriticConveyorBeltResults/'
else:
  ROOTDIR      = '/data/public/rw/pass.port//conveyor_belt/'

def train_gps(n_data,is_init_guide_train,n_trial):
  setup_save_dirs( ROOTDIR+'place_gps/',n_data,n_trial)
  if n_trial == -1:
    n_trial = determine_trial(ROOTDIR+'place_gps/n_data_'+str(n_data))
  guide_s,guide_a = load_guidance_data( parent_dir=ROOTDIR+'place_gps/', \
                                   proc_train_data_dir='processed_train_data',\
                                   n_data=n_data,n_trial=n_trial)
  key_configs = load_key_configs()
  dim_state  = (957,2,1)
  dim_action = 6
 
  train_results_dir =  './place_gps/n_data_'+str(n_data)+'/n_trial_'\
                        +str(n_trial)+'/train_results/'
  n_traj = 5
  time_steps = 20
  n_plan_data = len(guide_s)
  session = tf.Session()
  policy  = PlaceGPS(session,\
                    dim_action= dim_action,\
                    dim_state = dim_state,\
                    n_plan_data = n_plan_data,\
                    n_traj=n_traj,\
                    time_steps=time_steps,\
                    save_folder=train_results_dir,\
                    key_configs=key_configs,\
                    x_scaler=None,\
                    c_scaler=None)
  try:
    policy.load_guided_weights()
  except:
    policy.guided_train( guide_s,guide_a )
    policy.load_guided_weights()
  
  # This is more like REINFORCE with worm start
  performance = []

  current_best_J = -np.inf
  file_names = []
  traj_list = []
  w_r=1.0
  for n_iter in range(100):
    for i in range(n_traj):
      problem = ConveyorBelt() # different "initial" state 
      traj = problem.execute_policy(policy,time_steps)
      traj_list.append(traj)
      problem.env.Destroy()
      RaveDestroy()
    avg_J = np.mean([np.sum(traj['r']) for traj in traj_list])
    if avg_J > current_best_J:
      current_best_J = avg_J
      theta_star     = '/policy_search_'+str(n_iter)+'_'+str(current_best_J)+'.h5'
      policy.save_weights( theta_star )
      w_r -= 0.1
    else:
      w_r +=0.1
    performance.append( [avg_J,n_iter] )
    print "Updating policy..."
    fin = open(train_results_dir+'/performance.txt','a')
    fin.write( str(avg_J) + ','+ str(n_iter)+'\n' )
    fin.close()
    policy.RL_train(traj_list,guide_s,guide_a,n_iter,theta_star,w_r)
  create_done_marker(train_results_dir)

def create_policy_and_scale_data(alg,train_results_dir,data,tau,explr_const,v):
  key_configs = pickle.load(open('./key_configs/key_configs.p','r'))
  print "N_data = ",data[0].shape
  #dim_state  = (data[0].shape[1],data[0].shape[2])
  dim_state = (957,2)
  dim_action = 3
  session = tf.Session()

  print train_results_dir
  print train_results_dir
  print '========'
  key_configs = pickle.load(open('./key_configs/key_configs.p','r'))
  x_scaler = StandardScaler()
  scaled_x = x_scaler.fit_transform( data[1] )

  data_ ={}
  data_['s'] = data[0]
  data_['a'] = scaled_x
  data_['r'] = data[2]
  data_['sprime'] = data[3]
  data_['sumr'] = data[4]
  data_['traj_lengths'] = data[6]

  if alg=='soap':
    policy = PlaceSOAP(session,
                     dim_action,
                     dim_state,
                     key_configs=key_configs,\
                     x_scaler=x_scaler,\
                     tau=tau,\
                     save_folder=train_results_dir,\
                     explr_const=explr_const,\
                     visualize=v)  
  elif alg=='adq':
    policy = PlaceAdQ(session,
                     dim_action,
                     dim_state,
                     key_configs=key_configs,\
                     x_scaler=x_scaler,\
                     tau=tau,\
                     save_folder=train_results_dir,\
                     explr_const=explr_const,
                     visualize=v)
  elif alg.find('ddpg')!=-1:
    assert tau is not None, 'ddpg requires tau'
    policy = PlaceDDPG(session,
                     dim_action,
                     dim_state,
                     key_configs=key_configs,\
                     x_scaler=x_scaler,\
                     tau=tau,\
                     save_folder=train_results_dir,\
                     explr_const = explr_const,\
                     visualize=v)  
  elif alg == 'trpo':
    policy = PlaceTRPO(session,
                     dim_action,
                     dim_state,
                     key_configs=key_configs,\
                     tau=tau,\
                     x_scaler=x_scaler,\
                     save_folder=train_results_dir,\
                     explr_const=explr_const,\
                     visualize=v)  
  elif alg == 'gail':
    policy = PlaceGAIL(session,
                     dim_action,
                     dim_state,
                     key_configs=key_configs,\
                     a_scaler=x_scaler,\
                     tau=tau,\
                     save_folder=train_results_dir,\
                     explr_const=explr_const,\
                     visualize=v)  
  return policy,data_

def determine_trial(parent_dir):
  trial_numbers = [int(ftrial.split('_')[-1]) for ftrial in os.listdir(parent_dir)]
  if len(trial_numbers)==0: 
    return 0
  return np.max(trial_numbers)+1

def train_adv(args):
  alg     = args.a
  n_data  = args.n_data
  n_trial = args.n_trial
  d_lr    = args.d_lr
  g_lr    = args.g_lr
  tau     = args.tau  # epsilon in TRPO, tau in DDPG, lambda in SOAP
  v       = args.v
  explr_const = args.explr_const
  n_score_train = args.n_score
  train_results_dir,scaler_dir = setup_save_dirs( ROOTDIR,alg,n_data,n_trial,\
                                                  d_lr,g_lr,tau,n_score_train,explr_const )
  alg_dir = ROOTDIR+'/place_'+alg
  data = load_place_RL_data(alg_dir,n_data)

  policy,data = create_policy_and_scale_data(alg,train_results_dir,data,tau,explr_const,v) 
  S=data['s']
  A=data['a']
  R=data['r']
  Sprime = data['sprime']
  sumR = data['sumr']
  traj_lengths = data['traj_lengths']
  if alg == 'soap':
    score = sumR
  elif alg=='adq' or alg.find('ddpg')!=-1 or alg=='trpo': 
    score = R

  print "Starting train"
  if alg == 'trpo':
    policy.train( S,\
                  A,\
                  R,\
                  Sprime,\
                  sumR, \
                  traj_lengths,\
                  epochs=300,\
                  d_lr=1e-3,g_lr=1e-4 )
  elif alg == 'gail':
    policy.train( S,\
                  A,\
                  epochs=300,\
                  d_lr=1e-3,g_lr=1e-4 )
  else:
    policy.train( S,\
                  A,\
                  score, \
                  Sprime,\
                  epochs=300,\
                  d_lr=1e-3,g_lr=1e-4 )
  policy.saveWeights(additional_name='_1_')
  create_done_marker(train_results_dir)

def create_done_marker(train_results_dir):
  fin = open(train_results_dir+'/done_train.txt','a')
  fin.write( 'dummy file to mark done\n' )
  fin.close()

def test_uniform_policy(visualize):
  pi = PlaceUnif(ROOTDIR+'/uniform/' )
  pi.evaluate(visualize)

def parse_args():
  parser = argparse.ArgumentParser(description='Process configurations')
  parser.add_argument('-n_data',type=int,default=100)
  parser.add_argument('-a',default='soap')
  parser.add_argument('-g',action='store_true')
  parser.add_argument('-n_trial',type=int,default=-1)
  parser.add_argument('-i',type=int,default=0)
  parser.add_argument('-v',action='store_true')
  parser.add_argument('-tau',type=float,default=1e-5)
  parser.add_argument('-d_lr',type=float,default=1e-3)
  parser.add_argument('-g_lr',type=float,default=1e-4)
  parser.add_argument('-n_score',type=int,default=5)
  parser.add_argument('-otherpi',default='uniform')
  parser.add_argument('-epoch',type=int,default=0)
  parser.add_argument('-explr_const',type=float,default=0.0)
  args = parser.parse_args()
  return args

def main():
  args = parse_args()
  if args.a == 'unif':
    test_uniform_policy(args.v)
  else:
    train_adv( args )

if __name__ == '__main__':
  main()
