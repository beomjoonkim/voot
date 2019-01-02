import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

sys.path.append('../plot_configs/')
from plot_configs import *


color_dict = pickle.load(open('../NAMO/color_dict.p','r')) 

def get_rewards(algo_name,data_ticks):
  rewards = []
  for n_data in data_ticks:
    dtick_rwd=[]
    root = './place_'+algo_name+'/'+'n_data_'+str(n_data)+'/'
    trial_dirs = os.listdir(root)
    
    for trial_dir in trial_dirs:
      try:
        test_dirs = os.listdir(root+'/'+trial_dir+'/test_results/')
      except OSError:
        print 'Skipping ',root+'/'+trial_dir
        continue
      for ftest in test_dirs:
        data = pickle.load(open( root+'/'+trial_dir+'/test_results/'+ftest))
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
    root = './place_'+algo_name+'/'+'n_data_'+str(n_data)+'/'
    trial_dirs = os.listdir(root)
    
    max_rwds = []
    for trial_dir in trial_dirs:
      fin = open(root+'/'+trial_dir+'/train_results/performance.txt','r')
      max_rwd = -np.inf
      for l in fin.readlines():
        rwd = float(l.split(',')[0])
        if rwd > max_rwd: max_rwd = rwd
      max_rwds.append(max_rwd)
    max_rwds.remove(min(max_rwds))
    rewards.append(max_rwds)
  return np.array(rewards).T



def main():
  data_ticks =  range(100,1100,100)
  
  for i,alg in enumerate( ['soap','ddpg']): 
    vals = get_policy_rewards(alg,data_ticks)
    sns.tsplot(vals,data_ticks,ci=95,condition=alg,color=color_dict.values()[i])
  plt.show()


if __name__ == '__main__':
  main()
