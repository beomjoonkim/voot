import os
import numpy as np
import pickle
from plotters.plot_rl_curve import savefig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import socket
import argparse 

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

def get_pick_place_eval_dirs(args):
  n_data = args.n_data
  if args.pi == 'soap':
    pick_dir  = "/data/public/rw/pass.port/NAMO/n_data_"+str(n_data)+\
                "/halfpick/adv/tau_2.0/dg_lr_0.0001_0.0001/n_score_1/architecture_0/"
    place_dir = "/data/public/rw/pass.port/NAMO/n_data_"+str(n_data)+\
                  "/halfplace/adv/tau_2.0/dg_lr_0.0001_0.0001/n_score_1/architecture_4/"
    eval_dir = "/data/public/rw/pass.port/NAMO/n_data_"+str(n_data)+"/soap/adv/tau_2.0/dg_lr_0.0001_0.0001/n_score_1/architecture_1/explr_const_0.5/"
  elif args.pi == 'ddpg':
    pick_dir = "/data/public/rw/pass.port/NAMO/n_data_"+str(n_data)+\
              "/ddpg/adv/tau_0.001/dg_lr_0.0001_0.0001/n_score_1/architecture_0/"
    place_dir = "/data/public/rw/pass.port/NAMO/n_data_"+str(n_data)+\
              "/ddpg/adv/tau_0.001/dg_lr_0.0001_0.0001/n_score_1/architecture_0/"
    eval_dir = "/data/public/rw/pass.port/NAMO/n_data_"+str(n_data)+\
              "/ddpg/adv/tau_0.001/dg_lr_0.0001_0.0001/n_score_1/architecture_0/explr_const_0.5/"
  elif args.pi == 'trpo':
    pick_dir = "/data/public/rw/pass.port/NAMO/n_data_"+str(n_data)+"/trpo/adv/tau_0.3/dg_lr_0.0001_0.0001/n_score_1/architecture_0/"
    place_dir = "/data/public/rw/pass.port/NAMO/n_data_"+str(n_data)+"/trpo/adv/tau_0.3/dg_lr_0.0001_0.0001/n_score_1/architecture_0/"
    eval_dir = pick_dir+'/explr_const_0.5/'
  else:
    print "Not set yet"
    sys.exit(-1)

  return pick_dir,place_dir,eval_dir
args = parse_args()

DDPG_RESULT_DIR = "/data/public/rw/pass.port/NAMO/n_data_6000/ddpg/adv/tau_0.001/dg_lr_0.0001_0.0001/n_score_1/architecture_0/explr_const_0.5/n_trial_3/planner_result/"
UNIF_RESULT_DIR = '/data/public/rw/pass.port/NAMO/unif_planning//'
SOAP_RESULT_DIR = "/data/public/rw/pass.port/NAMO/n_data_2000/soap/adv/tau_2.0/dg_lr_0.0001_0.0001/n_score_1/architecture_1/explr_const_0.5/n_trial_1/planner_result//trial_1/" # add trial_1
GAIL_RESULT_DIR = "/data/public/rw/pass.port/NAMO/n_data_6000/gail/adv/tau_0.3/dg_lr_0.0001_0.0001/n_score_1/architecture_0/explr_const_0.5/n_trial_1/planner_result/"
TRPO_RESULT_DIR = "/data/public/rw/pass.port/NAMO/n_data_8000/trpo/adv/tau_0.3/dg_lr_0.0001_0.0001/n_score_1/architecture_0/explr_const_0.5/n_trial_0/planner_result/"

if socket.gethostname() == 'dell-XPS-15-9560':
  ROOTDIR      = '../../AdvActorCriticNAMOResults/'
else:
  ROOTDIR      = '/data/public/rw/pass.port//NAMO/'


t_lims = range(10,3500,10)
plt.figure()
color_dict = pickle.load(open('../NAMO/color_dict.p','r'))

all_t_vals = np.zeros((1,len(t_lims)))
all_t_vals[:] = 7.971962*0.95
sns.tsplot(all_t_vals,t_lims,ci=95,condition="95% Optimal",color=[1,0,1])
algo_list = ['soap','ddpg','unif','trpo','gail']
for algo in algo_list:
  all_t_vals = []
  if algo == 'unif':
    color_name = 'planner'
    algo = 'Unif'
    result_dir = UNIF_RESULT_DIR
  elif algo == 'ddpg':
    color_name = 'pick'
    algo = 'DDPG'
    result_dir = DDPG_RESULT_DIR
    #_,_,eval_dir = get_pick_place_eval_dirs(args)
    #result_dir = eval_dir+'/n_trial_'+str(args.n_trial)+'/planner_result/'
    #print result_dir
  elif algo == 'soap':
    color_name = 'soap'
    algo = 'AdMon'
    result_dir = SOAP_RESULT_DIR
    #if args.n_data == 2000 and args.n_trial == 1:
    #  result_dir = result_dir+'trial_1/'
  elif algo == 'gail':
    algo = 'GAIL'
    args.pi = 'gail'
    color_name = 'gan'
    result_dir = GAIL_RESULT_DIR
  elif algo=='trpo':
    color_name = 'place'
    args.pi='trpo'  
    algo = 'PPO'
    #_,_,eval_dir = get_pick_place_eval_dirs(args)
    #result_dir = eval_dir+'/n_trial_'+str(args.n_trial)+'/planner_result/'
    result_dir = TRPO_RESULT_DIR

  print algo,len(os.listdir(result_dir))
  VS_NODE_PLOT = False
  for result_file in os.listdir(result_dir):
    t_vals = []
    if result_file.find('.pkl')==-1: continue
    ns = np.array( pickle.load(open(result_dir+result_file,'r'))['rwd_time_list'] )
    solved_idxs = ns[:,2] == 0
    ns[solved_idxs,1] = 10
    if VS_NODE_PLOT:
      for i in range(1,51):
        t_vals.append( np.max(ns[0:i,1]))
    else:
      for t in t_lims:
        idxs = ns[:,0]<t
        if idxs.sum() == 0: continue
        t_vals.append( np.max(ns[idxs,1]) )
      if len(t_vals)!=len(t_lims):
        t_vals.append(t_vals[-1])
    all_t_vals.append(t_vals)
  all_t_vals = np.array(all_t_vals)

  if VS_NODE_PLOT:
    sns.tsplot(all_t_vals,t_lims,ci=95,condition=algo,color=color_dict[color_name])
  else:
    sns.tsplot(all_t_vals,range(1,51),ci=95,condition=algo,color=color_dict[color_name])
savefig('Time (s)','Average performance',fname=ROOTDIR+'/planning_result_' \
            +str(args.n_data)+'_'+str(args.n_trial))
