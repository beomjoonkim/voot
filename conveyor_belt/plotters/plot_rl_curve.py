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
  ROOTDIR      = '/data/public/rw/pass.port//conveyor_belt/'

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
    
    all_trials_rwds=[]
    all_trials_stds=[]
    for trial_dir in trial_dirs:
      if algo_name == 'gail' and n_data ==5000 and (trial_dir=='n_trial_2'):
        continue
      rwds = []
      rwds_stds = []
      try:
        fin = open(root+'/'+trial_dir+'/train_results/performance.txt','r')
      except:
        continue
      max_rwd = -np.inf
      for l in fin.readlines():
        if len(l.split(',')) == 3:
          rwd = float(l.split(',')[1])  
          r_std = float(l.split(',')[-1])
        else:
          rwd = float(l.split(',')[0])  
          r_std = float(l.split(',')[-1])
        if rwd>max_rwd:
          rwds.append(rwd)
          rwds_stds.append(r_std)
          max_rwd = rwd
        else:
          rwds.append(max_rwd)
          rwds_stds.append(r_std)
      print len(rwds)
      while len(rwds)<299:
        rwds.append(rwds[-1])
      all_trials_rwds.append(rwds)
      all_trials_stds.append(rwds_stds)
  return np.array(all_trials_rwds)


def savefig(xlabel,ylabel,fname=''):
  plt.legend(loc='best',prop={'size': 13})
  plt.xlabel(xlabel,fontsize=14,fontweight='bold')
  plt.ylabel(ylabel,fontsize=14,fontweight='bold')
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  print 'Saving figure ',fname+'.png'
  plt.savefig(fname+'.png',dpi=100,format='png') 

def plot():
  data_ticks =  range(299)
  data_ticks = np.array(data_ticks)*20*5+100
  print data_ticks
  for i,alg in enumerate( ['soap','trpo','ddpg_new','gail']): 
    print alg
    vals_fname = './plotters/'+alg+'_rl_curve.pkl'
    #if os.path.isfile( vals_fname ):
    #  vals = pickle.load(open(vals_fname,'r'))
    #else:
    vals = get_policy_rewards(alg,[int(sys.argv[1])])
    #  pickle.dump(vals,open(vals_fname,'wb'))
    #print [np.mean(v) for v in vals.T]
    alg_name = alg
    if alg == 'soap':
      color_name = 'soap'
      alg_name = 'AdMon'
    elif alg.find('ddpg')!=-1:
      color_name = 'pick'
      alg_name = 'DDPG'
    elif alg=='trpo':
      color_name = 'place'
      alg_name = 'PPO'
    elif alg=='gail':
      color_name = 'gan'
      alg_name = 'GAIL'
    print alg_name
    try:
      sns.tsplot(vals,data_ticks,ci=90,condition=alg_name,color=color_dict[color_name])
    except:
      import pdb;pdb.set_trace()

  plt.xlim(xmin=-1000)
  plt.ylim(ymax=4.5)
  savefig('Number of steps','Average rewards',fname=ROOTDIR+'/conv_belt_rl')

def main():
  plot()

if __name__ == '__main__':
  main()
