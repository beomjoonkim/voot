import os
import numpy as np
import pickle
from plotters.plot_rl_curve import savefig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import socket

SOAP_RESULT_DIR = '/data/public/rw/pass.port/conveyor_belt/n_data_4000/soap/dg_lr_0.001_0.0001/tau_2.0/explr_const_0.0/n_score_5/n_trial_3/planner_result/'
SOAP_RESULT_DIR = '/data/public/rw/pass.port/conveyor_belt/n_data_4000/soap/dg_lr_0.001_0.0001/tau_2.0/explr_const_0.0/n_score_5/n_trial_2/planner_result/'
UNIF_RESULT_DIR = '/data/public/rw/pass.port/conveyor_belt/unif_planning/'
GAIL_RESULT_DIR = '/data/public/rw/pass.port/conveyor_belt/n_data_5000/gail/dg_lr_0.001_0.0001/tau_0.2/explr_const_0.5/n_score_5/n_trial_0/planner_result/'
DDPG_RESULT_DIR = '/data/public/rw/pass.port/conveyor_belt/n_data_5000/ddpg_new/dg_lr_0.001_0.0001/tau_0.001/explr_const_0.5/n_score_5/n_trial_2/planner_result/'
TRPO_RESULT_DIR = '/data/public/rw/pass.port/conveyor_belt/n_data_5000/trpo/dg_lr_0.001_0.0001/tau_0.3/explr_const_0.5/n_score_5/n_trial_0/planner_result/'

if socket.gethostname() == 'dell-XPS-15-9560':
  ROOTDIR      = '../../AdvActorCriticNAMOResults/'
else:
  ROOTDIR      = '/data/public/rw/pass.port//conveyor_belt/'
  

t_lims = range(10,400,10)
plt.figure()
color_dict = pickle.load(open('../NAMO/color_dict.p','r'))

all_t_vals = np.zeros((1,len(t_lims)))
all_t_vals[:] = 4.62*0.95
sns.tsplot(all_t_vals,t_lims,ci=95,condition="95% Opt",color=[1,0,1])

for algo in ['unif','soap','gail','trpo','ddpg']:
  all_t_vals = []
  if algo == 'unif':
    color_name = 'planner'
    algo = 'Unif'
    result_dir = UNIF_RESULT_DIR
  elif algo == 'soap':
    color_name = 'soap'
    algo = 'AdMon'
    result_dir = SOAP_RESULT_DIR
  elif algo == 'gail':
    color_name = 'gan'
    algo = 'GAIL'
    result_dir = GAIL_RESULT_DIR
  elif algo == 'ddpg':
    color_name = 'pick'
    algo = 'DDPG'
    result_dir = DDPG_RESULT_DIR
  elif algo == 'trpo':
    color_name = 'place'
    algo = 'PPO'
    result_dir = TRPO_RESULT_DIR

  for result_file in os.listdir(result_dir):
    t_vals = []
    if result_file.find('.pkl')==-1: continue
    ns =np.array( pickle.load(open(result_dir+result_file,'r'))['rwd_time_list'] )
    for t in t_lims:
      idxs = ns[:,0]<t
      if idxs.sum() == 0: continue
      t_vals.append( np.max(ns[idxs,1]) )
    all_t_vals.append(t_vals)
  all_t_vals = np.array(all_t_vals)
  plot = sns.tsplot(all_t_vals,t_lims,ci=95,condition=algo,color=color_dict[color_name])
  ax = plot.axes
  ax.set_ylim(0,4.62*0.95)
  #sns.xlim(0,5)
savefig('Time (s)','Average rewards',fname=ROOTDIR+'/conv_belt_planning_result')
