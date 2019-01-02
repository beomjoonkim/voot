import numpy as np
import matplotlib
matplotlib.use('Agg')
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
import socket

#sys.path.append('../plot_configs/')
#from plot_configs import *


color_dict = pickle.load(open('../NAMO/color_dict.p','r')) 
if socket.gethostname() == 'dell-XPS-15-9560':
  ROOTDIR      = '../../AdvActorCriticConveyorBeltResults/'
else:
  ROOTDIR      = '/data/public/rw/pass.port//NAMO/'

def get_rewards(algo_name,data_ticks):
  rewards = []
  for n_data in data_ticks:
    dtick_rwd=[]
    root = ROOTDIR+'/place_'+algo_name+'/'+'n_data_'+str(n_data)+'/'
    trial_dirs = os.listdir(root)
    
    for trial_dir in trial_dirs:
      try:
        test_dirs = os.listdir(root+'/'+trial_dir+'/test_results/')
      except OSError:
        print 'Skipping ',root+'/'+trial_dir
        continue
      for ftest in test_dirs:
        data = pickle.load(open( root+'/'+trial_dir+'/test_results/'+ftest))
        if algo_name == 'trpo':
          dtick_rwd.append(data[1])
        else:
          dtick_rwd.append(data[0])
      if len(dtick_rwd)==200:break
      print root
      print len(dtick_rwd)
    rewards.append(dtick_rwd)
  return np.array(rewards).T


def get_policy_rewards(algo_name,data_ticks):
  rewards = []
  for n_data in data_ticks:
    dtick_rwd=[]
    root = ROOTDIR+'/place_'+algo_name+'/'+'n_data_'+str(n_data)+'/'
    trial_dirs = os.listdir(root)
    
    max_rwds = []
    for trial_dir in trial_dirs:
      try:
        fin = open(root+'/'+trial_dir+'/train_results/performance.txt','r')
      except:
        continue
      max_rwd = -np.inf
      for l in fin.readlines():
        if len(l.split(',')) == 3:
          rwd = float(l.split(',')[1])  
        else:
          rwd = float(l.split(',')[0])  
        """
        if algo_name == 'trpo':
          if len(l.split(',')) != 3:
            continue
          rwd = float(l.split(',')[1])  
        else:
          rwd = float(l.split(',')[0])
        """
        if rwd > max_rwd: max_rwd = rwd
      max_rwds.append(max_rwd)
    #max_rwds.remove(min(max_rwds))
    rewards.append(max_rwds)
    print n_data
    print rewards
  return rewards
  #return np.array(rewards).T

def main():
  data_ticks =  range(100,900,100)
  #for i,alg in enumerate( ['soap','ddpg','trpo']): 
  for i,alg in enumerate( ['soap','ddpg','trpo']): 
    print alg
    vals = get_policy_rewards(alg,data_ticks)
    print [np.mean(v) for v in vals]
    #sns.tsplot(vals,data_ticks,ci=95,condition=alg,color=color_dict.values()[i])
  plt.savefig(ROOTDIR+'/rwd_vs_data.png')


if __name__ == '__main__':
  main()
