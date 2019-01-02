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

#sys.path.append('../plot_configs/')
#from plot_configs import *


color_dict = pickle.load(open('../NAMO/color_dict.p','r')) 
if socket.gethostname() == 'dell-XPS-15-9560':
  ROOTDIR      = '../../AdvActorCriticConveyorBeltResults/'
else:
  ROOTDIR      = '/data/public/rw/pass.port//conveyor_belt/'

def get_policy_rewards(algo_name,data_ticks):
  rewards = []
  if algo_name == 'soap':
    tau = 2.0
    explr_const = 0.0
  elif algo_name == 'gail':
    tau = 0.2
    explr_const = 0.5
  elif algo_name == 'trpo':
    tau = 0.3
    explr_const = 0.5
  elif algo_name.find('ddpg')!=-1:
    tau = 0.001
    explr_const = 0.5
    
  for n_data in data_ticks:
    dtick_rwd=[]
    root = ROOTDIR+'/n_data_'+str(n_data)+'/'+algo_name+'/'+'dg_lr_0.001_0.0001/'+'tau_'+str(tau)\
          + '/explr_const_'+str(explr_const) + '/n_score_5/'
    trial_dirs = os.listdir(root)
    
    max_rwds = []
    for trial_dir in trial_dirs:
      if algo_name == 'gail' and n_data ==5000 and (trial_dir=='n_trial_2'):
        continue
      print algo_name,n_data,trial_dir
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
    #while len(max_rwds)>4:
    #  max_rwds.remove(min(max_rwds))
    #n_needed = float(len(max_rwds)-4)
    #mean_val = np.mean(max_rwds)
    #while n_needed < 0:
    #  max_rwds.append( np.mean(max_rwds)/abs(n_needed) )
    #  n_needed = float(len(max_rwds)-4)
    print algo_name,n_data
    rewards.append(np.sort(max_rwds)[::-1])
    print n_data
    print rewards
  return np.array(rewards).T


def plot():
  data_ticks =  range(100,900,100)
  data_ticks = [100,400,900]
  data_ticks.append(1500)
  data_ticks.append(2000)
  data_ticks.append(3000)
  data_ticks.append(4000)
  data_ticks.append(5000)
  plt.figure(1)
  plt.figure(2)

  n_episodes = np.array(data_ticks) / 50.0
  for i,alg in enumerate( ['soap','trpo','ddpg_new','gail']): 
    print alg
    vals_fname = './plotters/'+alg+'_learning_curve.pkl'
    if socket.gethostname() == 'dell-XPS-15-9560':
      vals = pickle.load(open(vals_fname,'r'))
    else:
      vals = get_policy_rewards(alg,data_ticks)
      print vals_fname
    print [np.mean(v) for v in vals.T]

    alg_name = alg
    if alg == 'soap':
      color_name = 'soap'
      alg_name = 'AdMon'
    elif alg.find('ddpg')!=-1:
      color_name = 'pick'
      alg_name='DDPG'
    elif alg=='trpo':
      color_name = 'place'
      alg_name='PPO'
    elif alg=='gail':
      color_name = 'gan'
      alg_name = 'GAIL'
    pickle.dump(vals,open(vals_fname,'wb'))
    plt.figure(1)
    if alg!='soap' and alg!='trpo':
      avgs = np.array([np.mean(v) for v in vals])
      stds = np.array([np.std(v) for v in vals])
    else:
      avgs = vals.mean(axis=0)
      stds = vals.std(axis=0)
    plt.errorbar(n_episodes,avgs,yerr=stds*1.96/2.,\
                 marker='o',label=alg_name,color=color_dict[color_name])
    #plt.figure(2)
    #sns.tsplot(vals,n_episodes,ci=95,condition=alg_name,color=color_dict[color_name])
  plt.figure(1)
  plt.ylim(ymax=4.5)
  savefig('Number of planning episodes','Average rewards',fname=ROOTDIR+'/conv_belt_max')
  """
  plt.figure(2)
  plt.ylim(1,4.5)
  savefig('Number of planning episodes','Average rewards',fname=ROOTDIR+'/conv_belt_avg')
  """
def main():
  plot()

if __name__ == '__main__':
  main()
