import socket
import os
import sys
import numpy as np


def get_trial_max( eval_dir ):
  try:
    fper = open(eval_dir+'/performance.txt','r')
  except:
    return None,None
  max_score = -np.inf
  last_epoch = -np.inf
  for ln in fper.readlines():
    epoch = int(ln.split(',')[0])
    score = float(ln.split(',')[1])
    if max_score < score:
      max_score = score
      max_epoch = epoch
  print eval_dir,max_score,max_epoch
  return max_score,max_epoch

def get_trial_vals( score_dir):
  trial_vals =[]
  n_trials = 0
  for trial in os.listdir(score_dir):
    trial_dir = score_dir+trial+'/'
    eval_dir = trial_dir+'/eval_results/uniform/'
    trial_max,last_epoch = get_trial_max( eval_dir+'/')
    if trial_max is not None:
      trial_vals.append([float(trial_max),int(last_epoch)])
    n_trials += 1
  return np.array(trial_vals),n_trials

def main():
  if socket.gethostname() == 'dell-XPS-15-9560':
    parent_dir      = '../../AdvActorCriticNAMOResults/'
  else:
    parent_dir      = '/data/public/rw/pass.port//NAMO/'

  n_data = sys.argv[1]
  target_dir = parent_dir+ '/n_data_' + n_data +'/'
  for config_dir in os.listdir( target_dir ):
    pi_dir = target_dir+config_dir+'/'
    for algo in os.listdir(pi_dir):
      if algo.find('scalers')!=-1: continue
      algo_dir = pi_dir+algo+'/'
      for lr in os.listdir( algo_dir ):
        lr_dir = algo_dir + lr +'/'
        for nscore in os.listdir(lr_dir):
          score_dir = lr_dir + nscore+'/'
          trial_vals,n_trials = get_trial_vals( score_dir )
          if len(trial_vals) > 0:
            print "Score dir",score_dir
            print "Mean,std,n_trials %f,%f,%d"%(np.mean(trial_vals[:,0]),np.std(trial_vals[:,0]),n_trials)
            print "Max,Min %f,%f"%(np.max(trial_vals[:,0]),np.min(trial_vals[:,0]))

if __name__ == '__main__':
  main()


