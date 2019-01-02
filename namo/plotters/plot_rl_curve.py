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
import subprocess

#sys.path.append('../plot_configs/')
#from plot_configs import *


color_dict = pickle.load(open('../NAMO/color_dict.p','r')) 
if socket.gethostname() == 'dell-XPS-15-9560':
  ROOTDIR      = '../../AdvActorCriticNAMOResults/'
else:
  ROOTDIR      = '/data/public/rw/pass.port//NAMO/'

def get_policy_rewards(algo_name,n_data):
  rewards = []
    
  all_trials_rwds=[]
  all_trials_stds=[]
  if n_data == 4000  and algo_name=='trpo':
    n_trials = [0,2]
  elif n_data==9000 and algo_name == 'ddpg':
    n_trials = [0,1,5,3]
  else:
    n_trials = [0,1,2,3]
  
  for n_trial in n_trials:
    rwds = []
    rwds_stds = []
    out = subprocess.Popen(['./'+algo_name+'_check_results.sh',str(n_data),str(n_trial)],stdout=subprocess.PIPE)
    fin = out.communicate()[0]
    max_rwd = -np.inf
    for line in fin.split('\n'):
      if len(line.split(','))==3:
        rwd = float(line.split(',')[1])
        r_std  = float(line.split(',')[2])
        epoch      = int(line.split(',')[0])
      elif len(line.split(','))==2 :
        rwd = float(line.split(',')[0])
        r_std  = float(line.split(',')[1])
      else:
        continue
      if rwd>max_rwd:
        rwds.append(rwd)
        rwds_stds.append( r_std )
        max_rwd = rwd
      else:
        rwds.append(max_rwd)
        rwds_stds.append( r_std )
    all_trials_rwds.append(rwds)
    all_trials_stds.append(rwds_stds)
  for trial_rwds in all_trials_rwds:
    while len(trial_rwds)>200:
      del trial_rwds[-1]
    while len(trial_rwds)<200:
      trial_rwds.append(trial_rwds[-1])
  return np.array(all_trials_rwds)


def savefig(xlabel,ylabel,fname=''):
  plt.legend(loc='best',prop={'size': 13})
  plt.xlabel(xlabel,fontsize=14,fontweight='bold')
  plt.ylabel(ylabel,fontsize=14,fontweight='bold')
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.savefig(fname+'.png',dpi=100,format='png') 

def plot():
  data_ticks =  range(200)
  for i,alg in enumerate( ['trpo','ddpg','admon','gail']): 
    print alg
    if socket.gethostname() == 'dell-XPS-15-9560':
      vals = pickle.load(open(vals_fname,'r'))
    else:
      vals = get_policy_rewards(alg,int(sys.argv[1]))
    print [np.mean(v) for v in vals.T]

    alg_name = alg
    if alg == 'admon':
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
    min_size = min( [len(vals[i]) for i in range(len(vals))])
    for i in range(len(vals)):
      vals[i] = vals[i][:min_size]
    vals = np.vstack(vals)
    sns.tsplot(vals,np.array(data_ticks)*50,ci=90,condition=alg_name,color=color_dict[color_name])
    """
    try:
      sns.tsplot(vals[:,1:],np.array(data_ticks)*20*5,ci=95,condition=alg_name,color=color_dict[color_name])
    except:
      print [len(vals[i]) for i in range(4)]
      import pdb;pdb.set_trace()
      print "error,",alg_name
    """

  savefig('Number of steps','Average rewards',fname=ROOTDIR+'/NAMO_rl_n_plan_exp_'+sys.argv[1])

def main():
  plot()

if __name__ == '__main__':
  main()
