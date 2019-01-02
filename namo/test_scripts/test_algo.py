from data_load_utils import load_RL_data
from NAMO_env import NAMO
from train_test_utils import *

import pickle
import os
import sys
import time
import socket
from multiprocessing.dummy import Pool as ThreadPool 
from plotters.get_max import get_max

import numpy as np
import tensorflow as tf

def create_epoch_done_marker(eval_dir,epoch):
  fin = open(eval_dir+'/'+str(epoch)+'_done','w')
  fin.write( 'dummy file to mark done\n' )
  fin.close()

def is_epoch_already_done( eval_dir,epoch ):
  if os.path.isfile(eval_dir+'/'+str(epoch)+'_done'):
    return True
  return False

def test_epoch(pi,eval_dir,epoch,visualize):
  #if is_epoch_already_done( eval_dir,epoch ):
  #  print "already done"
  #  sys.exit(-1)
  stime=time.time()
  #traj_list = pi.parallel_rollout()
  traj_list = pi.serial_rollout(visualize,n_insts=1)

  avg_reward = pi.record_performance( traj_list,epoch)
  print 'Test time',time.time()-stime
  print 'Score of this policy',avg_reward
  create_epoch_done_marker(eval_dir,epoch)

def find_best_weights( args,data_dir ):
  n_data  = args.n_data

  if n_data == 1000:
    trial = 3
  elif n_data == 2000:
    trial = 3
  elif n_data == 3000:
    trial = 1
  elif n_data == 4000:
    trial = 1
  elif n_data == 5000:
    trial = 2
  elif n_data == 6000:
    trial = 3
  elif n_data == 7000:
    trial = 3
  elif n_data == 8000:
    trial = 2
  elif n_data == 9000:
    trial = 3
  elif n_data == 10000:
    trial = 2

  eval_file = data_dir+"/explr_const_0.5/n_trial_"+\
                    str(trial)+"/eval_results/performance_with_noise.txt"
  best_score,_,_,best_epoch = get_max(eval_file)
  weight_dir = data_dir+'/n_trial_'+str(trial)+'/train_results/'
  print best_score

  pick_wfile = weight_dir+'a_gen_pick_epoch_'+str(best_epoch)+'.h5'
  place_wfile = weight_dir+'a_gen_place_epoch_'+str(best_epoch)+'.h5'
  return pick_wfile,place_wfile,trial

def main():
  args = parse_args() 
  n_data = args.n_data
  if args.pi.find('trpo')!=-1:
    """
    if n_data == 9000 or n_data == 6000 or n_data ==4000:
      args.pi = 'trpo'
    else:
      args.pi = 'trpo_d'
    """
    args.d_lr = 1e-4
    args.g_lr = 1e-4
    args.tau  = 0.3
    args.explr_const = 0.5
    args.architecture = 0
    eval_dir = '/data/public/rw/pass.port/NAMO/n_data_'\
              +str(n_data)\
              +'/'+args.pi+'/adv/tau_0.3/dg_lr_0.0001_0.0001/n_score_1/architecture_0//explr_const_0.5/'
    data_dir = '/data/public/rw/pass.port/NAMO/n_data_'+\
               str(n_data)+'/'+args.pi+'/adv/tau_0.3/dg_lr_0.0001_0.0001/n_score_1/architecture_0/'
    args.epoch = 0
  else:
    raise NotImplemented

  _,scaler_dir,_,parent_dir = setup_dirs(args)
  session              = tf.Session()
  policy               = create_pi(session,\
                                   '',\
                                   '',\
                                   args.pi,\
                                   args.Qloss,\
                                   float(args.d_lr),\
                                   float(args.g_lr),\
                                   float(args.tau),\
                                   int(args.n_score),\
                                   float(args.explr_const),args.architecture)
  # get the n_data

  # get the best trial's value
  pick_best_wfile,place_best_wfile,best_trial = find_best_weights(args,data_dir)
  policy.pick_pi.load_weights(pick_best_wfile)
  policy.place_pi.load_weights(place_best_wfile)
  policy.eval_dir = eval_dir

  scaler_dir = '/data/public/rw/pass.port/NAMO/n_data_'+str(n_data)+'/'+args.pi+'/adv/tau_0.3/dg_lr_0.0001_0.0001/n_score_1/architecture_0/n_trial_'+str(best_trial)+'/scalers/'
  policy.load_scalers(scaler_dir)
  test_epoch( policy, eval_dir, args.epoch, args.v )

if __name__ == '__main__':
  main()


