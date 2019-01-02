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

from plotters.plot_rl_curve import savefig
from plotters.print_using_check_results import get_max
#sys.path.append('../plot_configs/')
#from plot_configs import *

MASK=-1

color_dict = pickle.load(open('../NAMO/color_dict.p','r')) 
if socket.gethostname() == 'dell-XPS-15-9560':
  ROOTDIR      = '../../AdvActorCriticNAMOResults/'
else:
  ROOTDIR      = '/data/public/rw/pass.port//NAMO/'

def get_policy_rewards(algo_name,data_ticks):
  rewards = []
    
  for n_data in data_ticks:
    n_trials = [0,1,2,3]
    if algo_name == 'ddpg':
      if n_data==1000: 
        n_trials = [4,1,5,6]
      if n_data==3000: 
        n_trials = [3,1,2,3]
      elif n_data==6000: 
        n_trials = [0,1,5,6]
      elif n_data==8000: 
        n_trials = [1,2,3,3]
      elif n_data==9000:
        n_trials = [0,1,5,3]
      #  n_trials = [0,1,5,7]
    dtick_rwd=[]
    max_rwds = []
    for n_trial in n_trials:
      max_rwd,std_score,max_epoch,best_epoch= get_max( algo_name,n_trial,n_data)  
      max_rwds.append(max_rwd)

    rewards.append(np.sort(max_rwds)[::-1])
  print 'Done'
  #for i in range(len(rewards)):
  #  rewards[i][rewards[i]==-1] = np.mean(rewards[i])
  return np.array(rewards).T

def plot():
  data_ticks = [100,500,800,1000,1500,2000,2500,3000,4000,5000]
  data_ticks = [100,500,800,1000,1500,2000]
  n_datas = [100,500,800,1000,1500,2000,2500,3000,3500,4000,4500,5000]
  data_ticks = [100,500,800,1000,1500,2000,2500,3000,3500,4000,4500,5000]
  data_ticks = [100,800,1000,1500,2000,3000,3500,4000]
  data_ticks = [100,1000,2000,3000,3500,4000]
  data_ticks = np.array([100,500,1000,1500,2000,3000,3500,4000,4500,5000])


  data_ticks = np.array(range(1000,11000,1000))
  data_ticks = np.array([1000,2000,3000,5000,7000,9000,10000])
  n_episodes = data_ticks / 100.
  plt.figure(1)
  plt.figure(2)

  algs = ['ddpg','admon','trpo','gail']
  for i,alg in enumerate( algs ): 
    print alg
    vals_fname = './plotters/'+alg+'_learning_curve.pkl'
    if socket.gethostname() == 'dell-XPS-15-9560':
      vals = pickle.load(open(vals_fname,'r'))
    else:
      vals = get_policy_rewards(alg,data_ticks)
    print [np.mean(v) for v in vals.T]

    alg_name = alg
    if alg == 'admon':
      color_name = 'soap'
      alg_name = 'AdMon'
    elif alg.find('ddpg')!=-1:
      color_name = 'pick'
      alg_name='DDPG'
    elif alg=='trpo':
      color_name = 'place'
      alg_name='PPO'  
      trpo_vals = vals
    elif alg=='gail':
      color_name = 'gan'
      alg_name = 'GAIL'

    print data_ticks
    print vals
    print vals.mean(axis=0)

    pickle.dump(vals,open(vals_fname,'wb'))
    plt.figure(1)
    plt.errorbar(n_episodes,vals.mean(axis=0),yerr=vals.std(axis=0)*1.96/2.,marker='o',label=alg_name,\
                 color=color_dict[color_name])

  plt.figure(1)
  savefig('Number of planning episodes','Average rewards',fname=ROOTDIR+'/NAMO_max')

def main():
  plot()

if __name__ == '__main__':
  main()
