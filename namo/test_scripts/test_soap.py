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
from plotters.get_max import get_max

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
    #print "already done"
    #sys.exit(-1)
  stime=time.time()
  #traj_list = pi.parallel_rollout()
  traj_list = pi.serial_rollout(visualize,n_insts=5,n_steps=10)
  avg_reward = pi.record_performance( traj_list,epoch)
  print 'Test time',time.time()-stime
  print 'Score of this policy',avg_reward
  create_epoch_done_marker(eval_dir,epoch)

def get_pick_place_dirs(args):
  n_data  = args.n_data
  n_trial = args.n_trial
  pick_dir  = "/data/public/rw/pass.port/NAMO/n_data_"+str(10000)+\
              "/halfpick/adv/tau_2.0/dg_lr_0.0001_0.0001/n_score_1/architecture_1/"
  pick_dir  = "/data/public/rw/pass.port/NAMO/n_data_"+str(args.n_data)+\
              "/halfpick/adv/tau_2.0/dg_lr_0.0001_0.0001/n_score_1/architecture_1/"
  pick_dir  = "/data/public/rw/pass.port/NAMO/n_data_"+str(args.n_data)+\
              "/halfpick/adv/tau_2.0/dg_lr_0.0001_0.0001/n_score_1/architecture_0/"
  place_dir =  "/data/public/rw/pass.port/NAMO/n_data_"+str(args.n_data)+\
                "/halfplace/adv/tau_2.0/dg_lr_0.0001_0.0001/n_score_1/architecture_4/"
  return pick_dir,place_dir

def find_best_weights( args ):
  n_data  = args.n_data
  n_trial = args.n_trial
  pick_dir,place_dir = get_pick_place_dirs(args)

  if n_data == 1000:
    pick_n_trial = 1
    place_n_trial = 0
  elif n_data == 2000:
    place_n_trial = 01
  elif n_data == 3000:
    pick_n_trial = 0
    place_n_trial = 1
  elif n_data == 4000:
    place_n_trial = 3
  elif n_data == 5000:
    pick_n_trial = 2
    place_n_trial = 2
  elif n_data == 6000:
    place_n_trial = 2
  elif n_data == 7000:
    pick_n_trial = 1
    place_n_trial = 2
  elif n_data == 8000:
    place_n_trial = 2
  elif n_data == 9000:
    place_n_trial = 0
  elif n_data == 10000:
    pick_n_trial = 1
    place_n_trial = 3

  pick_n_trial =  3
  #place_n_trial = 0
  #pick_n_trial  = args.n_trial
  #place_n_trial = args.n_trial
  print 'pick_n_trial,place_n_trial',pick_n_trial,place_n_trial

  pick_eval_file  = pick_dir+"explr_const_0.5/n_trial_"+\
                    str(pick_n_trial)+"/eval_results/performance_with_noise.txt"
  place_eval_file = place_dir+"explr_const_0.5/n_trial_"+\
                    str(place_n_trial)+"/eval_results/performance_with_noise.txt"
  pick_score,_,_,pick_best_epoch   = get_max(pick_eval_file)
  place_score,_,_,place_best_epoch = get_max(place_eval_file)
  print 'pick score,place_score',pick_score,place_score

  pick_weight_dir = pick_dir+'/n_trial_'+str(pick_n_trial)+'/train_results/'
  place_weight_dir = place_dir+'/n_trial_'+str(place_n_trial)+'/train_results/'

  print 'pick epoch, place epoch ', pick_best_epoch,place_best_epoch
  pick_wfile = [f for f in os.listdir(pick_weight_dir) \
                  if f.find('a_gen_pick_epoch_'+str(pick_best_epoch))!=-1][0]
  place_wfile = [f for f in os.listdir(place_weight_dir) \
                  if f.find('a_gen_place_epoch_'+str(place_best_epoch))!=-1][0]
  print pick_wfile
  print place_wfile

  return pick_weight_dir+pick_wfile,place_weight_dir+place_wfile

def set_scalers( args,policy ):
  n_data  = args.n_data
  n_trial = args.n_trial
  pick_dir,place_dir = get_pick_place_dirs(args)
#  pick_scaler_dir = pick_dir+'/n_trial_'+str(1)+'/scalers/'
#  place_scaler_dir = place_dir+'/n_trial_'+str(0)+'/scalers/'

  pick_scaler_dir = pick_dir+'/n_trial_'+str(n_trial)+'/scalers/'
  place_scaler_dir = place_dir+'/n_trial_'+str(n_trial)+'/scalers/'
  print "Setting pick place scalers"
  print pick_scaler_dir
  print place_scaler_dir
  policy.pick_pi.a_scaler     = pickle.load(open(pick_scaler_dir+'/pick_a_scaler.pkl','r'))
  policy.place_pi.a_scaler    = pickle.load(open(place_scaler_dir+'/place_a_scaler.pkl','r'))
  policy.pick_pi.misc_scaler  = pickle.load(open(pick_scaler_dir+'/pick_misc_scaler.pkl','r'))
  policy.place_pi.misc_scaler = pickle.load(open(place_scaler_dir+'/place_misc_scaler.pkl','r'))

def main():
  args = parse_args() 
  args.pi = 'soap'
  args.tau = 2.0
  args.explr_const = 0.5
  args.n_score = 1
  args.d_lr = 1e-4
  args.g_lr = 1e-4
  args.architecture = 1
  #args.n_trial = 'fixed_halfpick'
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
    pick_dir,place_dir = get_pick_place_dirs(args)
    pick_weight_dir = pick_dir+'/n_trial_'+str(args.n_trial)+'/train_results/'
    place_weight_dir = place_dir+'/n_trial_'+str(args.n_trial)+'/train_results/'
    pick_wfile = [f for f in os.listdir(pick_weight_dir) \
                    if f.find('a_gen_pick_epoch_'+str(args.epoch)+'_')!=-1][0]
    place_wfile = [f for f in os.listdir(place_weight_dir) \
                    if f.find('a_gen_place_epoch_'+str(args.epoch)+'_')!=-1][0]
    policy.pick_pi.a_gen.load_weights(pick_weight_dir+pick_wfile)
    policy.place_pi.a_gen.load_weights(place_weight_dir+place_wfile)
    set_scalers(args,policy)
    #policy.load_weights(args.epoch)
    #policy.load_scalers(scaler_dir)
  elif args.wpath == 'bestweights':
    pick_best_wfile,place_best_wfile = find_best_weights(args)
    policy.pick_pi.load_weights(pick_best_wfile)
    policy.place_pi.load_weights(place_best_wfile)
    set_scalers(args,policy)
    args.epoch = args.n_trial
  else:
    # to test this, make directories in the soap directory and store weights from
    # halfplace and halfpick
    policy.load_weights_from_path(args.wpath)
    policy.load_scalers(args.wpath)


  policy.eval_dir = eval_dir
  test_epoch( policy, eval_dir, args.epoch, args.v )

if __name__ == '__main__':
  main()


