from data_load_utils import load_RL_data
from NAMO_env import NAMO
from train_test_utils import *

import pickle
import os
import sys
import time
import socket
from multiprocessing.dummy import Pool as ThreadPool 


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
  if is_epoch_already_done( eval_dir,epoch ):
    print "already done"
    sys.exit(-1)
  stime=time.time()
  #traj_list = pi.parallel_rollout()
  traj_list = pi.serial_rollout(visualize)


  avg_reward = pi.record_performance( traj_list,epoch)
  print 'Test time',time.time()-stime
  print 'Score of this policy',avg_reward
  create_epoch_done_marker(eval_dir,epoch)

def main():
  args = parse_args() 
  args.pi = 'halfplace'
  args.tau = 2.0
  args.explr_const = 0.5
  args.n_score = 1
  args.d_lr = 1e-4
  args.g_lr = 1e-4
  args.architecture = 4
  print args.pi,args.tau,args.explr_const,args.d_lr,args.g_lr,args.architecture
  weight_dir,scaler_dir,eval_dir,parent_dir = setup_dirs(args)
  session              = tf.Session()
  policy               = create_pi(session,\
                                   weight_dir,\
                                   eval_dir,\
                                   args.pi,\
                                   args.Qloss,\
                                   float(args.d_lr),\
                                   float(args.g_lr),\
                                   float(args.tau),\
                                   int(args.n_score),\
                                   float(args.explr_const),args.architecture)
  if args.wpath == 'None':
    policy.load_weights(args.epoch)
    policy.load_scalers(scaler_dir)
  else:
    policy.load_weights_from_path(args.wpath)
    policy.load_scalers(args.wpath)
  test_epoch( policy, eval_dir, args.epoch, args.v )

if __name__ == '__main__':
  main()


