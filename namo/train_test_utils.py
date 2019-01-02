from generators.PickPlain import PickPlain
from generators.PickSOAP import PickSOAP
from generators.PlaceSOAP import PlaceSOAP
from generators.DDPG import DDPG
from generators.Uniform import UniformPlace,UniformPick
from generators.SOAP import SOAP
from generators.TRPO import TRPO
from generators.GAIL import GAIL
from generators.AdQ import AdQ
from generators.HalfPickSOAP import HalfPickSOAP
from generators.HalfPlaceSOAP import HalfPlaceSOAP

from data_load_utils       import load_RL_data,get_sars_data,get_data_dimensions
from NAMO_env import NAMO
from keras import backend as K
from openravepy import *

import sys
import os
import argparse
import numpy as np
import socket

def setup_dirs(args,):
  n_data       = args.n_data; 
  n_trial      = args.n_trial;
  pi_name      = args.pi
  Qloss        = args.Qloss
  tau          = args.tau
  otherpi_name = args.otherpi
  d_lr         = args.d_lr
  g_lr         = args.g_lr
  n_score      = args.n_score
  architecture  = args.architecture

  if socket.gethostname() == 'dell-XPS-15-9560':
    parent_dir      = '../../AdvActorCriticNAMOResults/'
  else:
    parent_dir      = '/data/public/rw/pass.port//NAMO/'
  performance_dir   = './'
  weight_dir,scaler_dir = setup_save_dirs( parent_dir,pi_name,otherpi_name,\
                                           Qloss,n_data,n_trial,d_lr,g_lr,tau,\
                                           n_score,architecture )
  eval_dir              = get_eval_dir( args,parent_dir  )
  return weight_dir,scaler_dir,eval_dir,parent_dir

def get_weight_and_scaler_dir( args,parent_dir ):
  n_data       = args.n_data; 
  n_trial      = args.n_trial;
  pi_name      = args.pi
  Qloss        = args.Qloss
  d_lr         = args.d_lr
  g_lr         = args.g_lr
  n_score      = args.n_score
  n_data_dir   = parent_dir +'/n_data_'+str(n_data)
  pi_dir       = n_data_dir  +'/'+pi_name
  Qloss_dir    = pi_dir    +'/'+Qloss
  lr_dir       = Qloss_dir    +'/dg_lr_'+str(d_lr)+'_'+str(g_lr)+'/'
  nscore_dir   = lr_dir    +'/'+'/n_score_'+str(n_score) +'/'
  trial_dir    = nscore_dir  + '/n_trial_' + str(n_trial)
  weight_dir   = trial_dir   + '/train_results/'
  scaler_dir   = pi_dir   + '/scalers/'
  return weight_dir,scaler_dir

def get_eval_dir( args,parent_dir):
  n_data_dir        = parent_dir +'/n_data_'+str(args.n_data)
  pi_dir            = n_data_dir +'/'+args.pi
  scaler_dir        = pi_dir +'/scalers/'
  Qloss_dir         = pi_dir +'/'+str(args.Qloss)
  tau_dir           = Qloss_dir+'/tau_'+str(args.tau)+'/'
  lr_dir            = tau_dir +'/dg_lr_'+str(args.d_lr)+'_'+str(args.g_lr)+'/'
  nscore_dir        = lr_dir+'/'+'/n_score_'+str(args.n_score) +'/'
  architecture_dir  = nscore_dir+'/'+'/architecture_'+str(args.architecture) +'/'
  explr_dir         = architecture_dir+'/explr_const_'+str(args.explr_const)
  trial_dir         = explr_dir + '/n_trial_' + str(args.n_trial)
  train_results_dir = trial_dir + '/train_results/'
  eval_dir  = trial_dir + '/eval_results/'
  if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)
  return eval_dir

def find_weight_file(weight_dir,epoch):
  for wfile in os.listdir(weight_dir):
    if wfile.find('h5') == -1: continue
    wfile_epoch = int(wfile.split('_')[-1].split('.')[0])
    if wfile_epoch == epoch:
      return wfile

def determine_n_trial_and_n_data_pairs(pi_dir):
  ndir_to_train    = range(100,1100,100)
  ndir_to_train    = [1000,5000]
  ntrials_to_train = range(4) 
  
  to_train = []
  for ndata in ndir_to_train:
    for trial in ntrials_to_train:
      train_results = './'+pi_dir+'/n_data_'+str(ndata)+'/n_trial_'\
                      +str(trial)+'/train_results/'  
      is_dir_exists = os.path.isdir(train_results)
      is_done_training = 'done_train.txt' in os.listdir(train_results) \
                          if is_dir_exists else False
      if not is_done_training:
        to_train.append([ndata,trial])
  return to_train

def setup_save_dirs(parent_dir,pi_name,otherpi_name,Qloss,\
                    n_data,n_trial,d_lr,g_lr,tau,nscore_train,architecture):
  n_data_dir        = parent_dir +'/n_data_'+str(n_data)
  pi_dir            = n_data_dir +'/'+pi_name
  Qloss_dir         = pi_dir +'/'+Qloss
  tau_dir           = Qloss_dir+'/tau_'+str(tau)+'/'
  lr_dir            = tau_dir +'/dg_lr_'+str(d_lr)+'_'+str(g_lr)+'/'
  nscore_dir        = lr_dir+'/'+'/n_score_'+str(nscore_train) +'/'
  architecture_dir  = nscore_dir+'/'+'/architecture_'+str(architecture) +'/'
  trial_dir         = architecture_dir + '/n_trial_' + str(n_trial)
  scaler_dir        = trial_dir +'/scalers/'
  train_results_dir = trial_dir + '/train_results/'
    
  if not os.path.exists(train_results_dir):
    os.makedirs( train_results_dir )
  if not os.path.exists(scaler_dir):
    os.mkdir( scaler_dir )
  return train_results_dir,scaler_dir

def load_data(n_data):
  pick_data,place_data = load_RL_data(n_data)
  return pick_data,place_data
 
def create_pi(session,\
              weight_dir,eval_dir,pi_name,\
              Qloss,d_lr,g_lr,tau,n_score,\
              explr_const,architecture):
  #dim_misc,dim_cvec,dim_pick,dim_place = get_data_dimensions(pick_data,place_data)
  dim_misc = 9
  dim_cvec = (1018,4)
  dim_pick = 6
  dim_place = 3
  key_configs = pickle.load(open('./key_configs/key_configs.p','r'))

  # d_lr_pick = 1e-4
  # g_lr_pick = 1e-4
  # d_lr_pick = 1e-3
  # g_lr_pick = 1e-4

  if pi_name.find('ddpg')!=-1:
    policy = DDPG(session,\
                 dim_pick,\
                 dim_place,\
                 dim_cvec,\
                 dim_misc,\
                 weight_dir,\
                 eval_dir,\
                 key_configs,\
                 Qloss,\
                 d_lr_pick=d_lr,\
                 g_lr_pick=g_lr,\
                 d_lr_place=d_lr,\
                 g_lr_place=g_lr,
                 tau_pick=tau,\
                 tau_place=tau,\
                 explr_const=explr_const)
  elif pi_name.find('trpo')!=-1:
    policy = TRPO(session,\
                 dim_pick,\
                 dim_place,\
                 dim_cvec,\
                 dim_misc,\
                 weight_dir,\
                 eval_dir,\
                 key_configs,\
                 Qloss,\
                 d_lr_pick=d_lr,\
                 g_lr_pick=g_lr,\
                 d_lr_place=d_lr,\
                 g_lr_place=g_lr,
                 tau_pick=tau,\
                 tau_place=tau,\
                 explr_const=explr_const,architecture=architecture)
  elif pi_name.find('gail')!=-1:
    policy = GAIL(session,\
                 dim_pick,\
                 dim_place,\
                 dim_cvec,\
                 dim_misc,\
                 weight_dir,\
                 eval_dir,\
                 key_configs,\
                 Qloss,\
                 d_lr_pick=d_lr,\
                 g_lr_pick=g_lr,\
                 d_lr_place=d_lr,\
                 g_lr_place=g_lr,
                 tau_pick=tau,\
                 tau_place=tau,\
                 explr_const=explr_const,\
                 architecture=architecture)
  elif pi_name.find('soap')!=-1:
    policy = SOAP(session,\
                 dim_pick,\
                 dim_place,\
                 dim_cvec,\
                 dim_misc,\
                 weight_dir,\
                 eval_dir,\
                 key_configs,\
                 Qloss,\
                 d_lr_pick=d_lr,\
                 g_lr_pick=g_lr,\
                 d_lr_place=d_lr,\
                 g_lr_place=g_lr,
                 tau_pick=tau,\
                 tau_place=tau,\
                 explr_const=explr_const,architecture=architecture)
  elif pi_name.find('adq')!=-1:
    policy = AdQ(session,\
                 dim_pick,\
                 dim_place,\
                 dim_cvec,\
                 dim_misc,\
                 weight_dir,\
                 eval_dir,\
                 key_configs,\
                 Qloss,\
                 d_lr_pick=d_lr,\
                 g_lr_pick=g_lr,\
                 d_lr_place=d_lr,\
                 g_lr_place=g_lr,
                 tau_pick=tau,\
                 tau_place=tau,\
                 explr_const=explr_const)
  elif pi_name.find('halfpick')!=-1:
    policy = HalfPickSOAP(session,\
                       dim_pick,\
                       dim_place,\
                       dim_cvec,\
                       dim_misc,\
                       weight_dir,\
                       eval_dir,\
                       key_configs,\
                       Qloss,\
                       d_lr_pick=d_lr,\
                       g_lr_pick=g_lr,\
                       d_lr_place=d_lr,\
                       g_lr_place=g_lr,
                       tau_pick=tau,\
                       tau_place=tau,\
                       explr_const=explr_const,\
                       architecture=architecture)
  elif pi_name.find('halfplace')!=-1:
    policy = HalfPlaceSOAP(session,\
                       dim_pick,\
                       dim_place,\
                       dim_cvec,\
                       dim_misc,\
                       weight_dir,\
                       eval_dir,\
                       key_configs,\
                       Qloss,\
                       d_lr_pick=d_lr,\
                       g_lr_pick=g_lr,\
                       d_lr_place=d_lr,\
                       g_lr_place=g_lr,
                       tau_pick=tau,\
                       tau_place=tau,\
                       explr_const=explr_const,\
                       architecture=architecture)
 
  else:
    print "Invalid algo name"
    sys.exit(-1)
  return policy

def get_best_weight_file( eval_dir,weight_dir ):
  fper = open(eval_dir+'/performance.txt','r')
  max_score = -np.inf
  last_epoch = -np.inf
  for ln in fper.readlines():
    epoch = int(ln.split(',')[0])
    score = float(ln.split(',')[1])
    if max_score < score:
      max_score = float(score)
      max_epoch = int(epoch)
  for fin in os.listdir(weight_dir):
    if fin.find('.h5')==-1: continue
    f_epoch = int(fin.split('_')[-1].split('.')[0])
    if f_epoch == max_epoch:
      return weight_dir+str(fin),max_score
  raise ValueError,'There must be a corresponding weight file'

def determine_best_weight( config_dir ):
  #TODO: Rewrite
  best_score = -np.inf
  for trial_dir in os.listdir(config_dir):
    eval_dir        = config_dir+'/'+trial_dir+'/'+'/eval_results/uniform/'
    weight_dir      = config_dir+'/'+trial_dir+'/'+'/train_results/'
    wfile,score = get_best_weight_file(eval_dir,weight_dir)
    if score > best_score:
      best_wfile = wfile
      best_score = score
  return best_wfile,best_score

def get_pi_name_from_dir(otherpi_dir):
  for kidx,k in enumerate(otherpi_dir.split('/')):
    if k.find('n_data_')!=-1:
      pi_name = otherpi_dir.split('/')[kidx+1]
      return pi_name

def train_policy(alg,pick_data,place_data,traj_data,policy,weight_dir,v): 
  print "Starting train"
  policy.train(pick_data,\
               place_data,\
               traj_data,
               n_epochs=300,\
               visualize=v)
  create_done_marker(weight_dir)

def create_done_marker(train_results_dir):
  fin = open(train_results_dir+'/done_train.txt','a')
  fin.write( 'dummy file to mark done\n' )
  fin.close()

def parse_args():
  parser = argparse.ArgumentParser(description='Process configurations')
  parser.add_argument('-n_data',type=int,default=100)
  parser.add_argument('-pi',default='soap')
  parser.add_argument('-g',action='store_true') # what's this?
  parser.add_argument('-n_trial',type=int,default=-1)
  parser.add_argument('-i',type=int,default=0)
  parser.add_argument('-pidx',type=int,default=0)
  parser.add_argument('-v',action='store_true')
  parser.add_argument('-tau',type=float,default=1e-5)
  parser.add_argument('-d_lr',type=float,default=1e-3)
  parser.add_argument('-g_lr',type=float,default=1e-4)
  parser.add_argument('-n_score',type=int,default=1)
  parser.add_argument('-Qloss',default='adv')
  parser.add_argument('-otherpi',default='uniform')
  parser.add_argument('-epoch',type=int,default=0)
  parser.add_argument('-explr_const',type=float,default=0.0)
  parser.add_argument('-wpath',default='None')
  parser.add_argument('-architecture',type=int,default=0)
  args = parser.parse_args()
  return args


