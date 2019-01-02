import pickle
import os
import sys
import time
import socket
import numpy as np

from train_scripts.train_algo import parse_args,create_policy
from data_load_utils import load_place_RL_data,load_key_configs, \
                            load_guidance_data,load_place_data,\
                            format_RL_data,setup_save_dirs

ROOTDIR = './'
if socket.gethostname() == 'dell-XPS-15-9560':
  ROOTDIR      = '../../AdvActorCriticConveyorBeltResults/'
else:
  ROOTDIR      = '/data/public/rw/pass.port//conveyor_belt/'

def record_rwd( epoch,eval_dir,avg_rwd,std_rwd ):
  pfile = open(eval_dir+'/performance.txt','a')
  pfile.write(str(epoch)+','+str(avg_rwd)+','+str(std_rwd)+'\n')
  pfile.close()

def create_epoch_done_marker(eval_dir,epoch):
  fin = open(eval_dir+'/'+str(epoch)+'_done','w')
  fin.write( 'dummy file to mark done\n' )
  fin.close()

def is_epoch_already_done( eval_dir,epoch ):
  if os.path.isfile(eval_dir+'/'+str(epoch)+'_done'):
    return True

def main():
  args = parse_args()
  alg     = args.a
  n_data  = args.n_data
  n_trial = args.n_trial
  d_lr    = args.d_lr
  g_lr    = args.g_lr
  tau     = args.tau  # epsilon in TRPO, tau in DDPG, lambda in SOAP
  v       = args.v
  epoch   = args.epoch
  explr_const = args.explr_const
  n_score_train = args.n_score
  train_results_dir,scaler_dir = setup_save_dirs( ROOTDIR,alg,n_data,n_trial,d_lr,\
                                                  g_lr,tau,n_score_train,explr_const )
  eval_dir = train_results_dir

  if is_epoch_already_done( eval_dir,epoch ):
    print "already done"
    sys.exit(-1)

  agen_wfname = '/a_gen_epoch_'+str(epoch)+'.h5'
  disc_wfname = '/disc_epoch_'+str(epoch)+'.h5'
  alg_dir = ROOTDIR+'/place_'+alg
  data = load_place_RL_data(alg_dir,n_data)
  policy,data = create_policy(alg,train_results_dir,data,tau,explr_const,v) 
  policy.load_weights(agen_wfname,disc_wfname)
  avg_J,std_J = policy.evaluate()

  record_rwd( epoch,eval_dir,avg_J,std_J )
  create_epoch_done_marker(eval_dir,epoch)

if __name__ == '__main__':
  main()


