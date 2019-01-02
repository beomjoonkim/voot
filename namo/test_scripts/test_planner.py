from planners.forward_search import forward_dfs_search
from NAMO_env import NAMO
from generators.Uniform import UniformPick,UniformPlace
from train_test_utils import *

import tensorflow as tf
import numpy as np

UNIF_RESULT = '/data/public/rw/pass.port/NAMO/unif_planning/'

SOAP_PICK_WPATH = "/data/public/rw/pass.port/NAMO/n_data_9000/halfpick/adv/tau_2.0/dg_lr_0.0001_0.0001/n_score_1/architecture_0/n_trial_0/train_results/a_gen_pick_epoch_61_0.8601547345514176_0.8538614308025224.h5"
SOAP_PLACE_WPATH = "/data/public/rw/pass.port/NAMO/n_data_9000/halfplace/adv/tau_2.0/dg_lr_0.0001_0.0001/n_score_1/architecture_4/n_trial_0/train_results/a_gen_place_epoch_61_2.0671639239681285_1.9936087766428578.h5"
SOAP_RESULT = "/data/public/rw/pass.port/NAMO/n_data_9000/soap/adv/tau_2.0/dg_lr_0.0001_0.0001/n_score_1/architecture_1/explr_const_0.5/planner_result/"

SOAP_PICK_WPATH="/data/public/rw/pass.port/NAMO/n_data_10000/halfpick/adv/tau_2.0/dg_lr_0.0001_0.0001/n_score_1/architecture_0/n_trial_3/train_results/a_gen_pick_epoch_41_0.850063484024439_0.8490532590249642.h5"
SOAP_PLACE_WPATH = "/data/public/rw/pass.port/NAMO/n_data_10000/halfplace/adv/tau_2.0/dg_lr_0.0001_0.0001/n_score_1/architecture_4/n_trial_3/train_results/a_gen_place_epoch_41_2.0294154590506546_1.9556612807407159.h5"
SOAP_RESULT = "/data/public/rw/pass.port/NAMO/n_data_10000/soap/adv/tau_2.0/dg_lr_0.0001_0.0001/n_score_1/architecture_1/explr_const_0.5/planner_result/"


SOAP_PICK_WPATH="/data/public/rw/pass.port/NAMO/n_data_10000/halfpick/adv/tau_2.0/dg_lr_0.0001_0.0001/n_score_1/architecture_0/n_trial_2/train_results/a_gen_pick_epoch_181_0.8799269711319161_0.8414218462763441.h5"
SOAP_PLACE_WPATH = "/data/public/rw/pass.port/NAMO/n_data_10000/halfplace/adv/tau_2.0/dg_lr_0.0001_0.0001/n_score_1/architecture_4/n_trial_2/train_results/a_gen_place_epoch_181_2.0498341491422836_1.9424335130235733.h5"
SOAP_RESULT = "/data/public/rw/pass.port/NAMO/n_data_10000/soap/adv/tau_2.0/dg_lr_0.0001_0.0001/n_score_1/architecture_1/explr_const_0.5/n_trial_2/planner_result/"


#TRPO_PICK_WPATH = "/data/public/rw/pass.port/NAMO/n_data_4000/trpo/adv/tau_0.3/dg_lr_0.0001_0.0001/n_score_1/architecture_0/n_trial_3/train_results/a_gen_pick_epoch_293.h5"
#TRPO_PLACE_WPATH = "/data/public/rw/pass.port/NAMO/n_data_4000/trpo/adv/tau_0.3/dg_lr_0.0001_0.0001/n_score_1/architecture_0/n_trial_3/train_results/a_gen_place_epoch_293.h5"
#TRPO_RESULT = "/data/public/rw/pass.port/NAMO/n_data_4000/trpo/adv/tau_0.3/dg_lr_0.0001_0.0001/n_score_1/architecture_0/explr_const_0.5/n_trial_3/planner_result/"

GAIL_WPATH = "/data/public/rw/pass.port/NAMO/n_data_5000/gail/dg_lr_0.001_0.0001/tau_0.2/explr_const_0.5/n_score_5/n_trial_0/train_results/a_genepoch_58_2.6.h5"
GAIL_RESULT = "/data/public/rw/pass.port/NAMO/n_data_5000/gail/dg_lr_0.001_0.0001/tau_0.2/explr_const_0.5/n_score_5/n_trial_0/planner_result/"

DDPG_PICK_WPATH = "/data/public/rw/pass.port/NAMO/n_data_9000/ddpg/adv/tau_0.001/dg_lr_0.0001_0.0001/n_score_1/architecture_0/n_trial_3/train_results/a_gen_pick_epoch_114.h5"
DDPG_PLACE_WPATH = "/data/public/rw/pass.port/NAMO/n_data_9000/ddpg/adv/tau_0.001/dg_lr_0.0001_0.0001/n_score_1/architecture_0/n_trial_3/train_results/a_gen_place_epoch_114.h5"
DDPG_RESULT = "/data/public/rw/pass.port/NAMO/n_data_9000/ddpg/adv/tau_0.001/dg_lr_0.0001_0.0001/n_score_1/architecture_0/explr_const_0.5/n_trial_3/planner_result/"


def create_policy(args,problem):
  alg = args.pi
  tau = args.tau
  explr_const = args.explr_const
  train_results_dir = ''

  dim_misc = 9
  dim_cvec = (1018,4)
  dim_pick = 6
  dim_place = 3
  key_configs = pickle.load(open('./key_configs/key_configs.p','r'))

  session = tf.Session()
  if alg.find('soap')!=-1:
    policy = SOAP(session,\
                 dim_pick,\
                 dim_place,\
                 dim_cvec,\
                 dim_misc,\
                 weight_dir='',\
                 eval_dir='',\
                 key_configs=key_configs,\
                 Qloss='adv',\
                 d_lr_pick=1e-4,\
                 g_lr_pick=1e-4,\
                 d_lr_place=1e-4,\
                 g_lr_place=1e-4,
                 tau_pick=2.0,\
                 tau_place=2.0,\
                 explr_const=explr_const,
                 architecture=1)
  elif alg.find('ddpg')!=-1:
    policy = DDPG(session,\
                 dim_pick,\
                 dim_place,\
                 dim_cvec,\
                 dim_misc,\
                 weight_dir="",\
                 eval_dir="",\
                 key_configs=key_configs,\
                 Qloss='adv',\
                 d_lr_pick=1e-4,\
                 g_lr_pick=1e-4,\
                 d_lr_place=1e-4,\
                 g_lr_place=1e-4,
                 tau_pick=0.001,\
                 tau_place=0.001,\
                 explr_const=0.5)
  elif alg.find('trpo')!=-1:
    policy = TRPO(session,\
                 dim_pick,\
                 dim_place,\
                 dim_cvec,\
                 dim_misc,\
                 "",\
                 "",\
                 key_configs,\
                 'adv',\
                 d_lr_pick=1e-4,\
                 g_lr_pick=1e-4,\
                 d_lr_place=1e-4,\
                 g_lr_place=1e-4,
                 tau_pick=0.3,\
                 tau_place=0.3,\
                 explr_const=0.5,\
                 architecture=0)
  elif alg.find('gail')!=-1:
    policy = GAIL(session,\
                 dim_pick,\
                 dim_place,\
                 dim_cvec,\
                 dim_misc,\
                 "",\
                 "",\
                 key_configs,\
                 "adv",\
                 d_lr_pick=1e-4,\
                 g_lr_pick=1e-4,\
                 d_lr_place=1e-4,\
                 g_lr_place=1e-4,
                 tau_pick=0.3,\
                 tau_place=0.3,\
                 explr_const=0.5,\
                 architecture=0)

  return policy

def load_weights_and_get_result_dir(args,policy):
  if args.pi.find('soap')!=-1:
    pick_weight_file = SOAP_PICK_WPATH
    place_weight_file = SOAP_PLACE_WPATH
    planner_result_dir = SOAP_RESULT
  elif args.pi.find( 'trpo' ) != -1:
    pick_weight_file = TRPO_PICK_WPATH
    place_weight_file = TRPO_PLACE_WPATH
    planner_result_dir = TRPO_RESULT
  elif args.pi.find( 'gail' ) != -1:
    planner_result_dir = GAIL_RESULT
    pi.a_gen.load_weights(GAIL_WPATH)
  elif args.pi.find( 'ddpg' ) != -1:
    pick_weight_file = DDPG_PICK_WPATH
    place_weight_file = DDPG_PLACE_WPATH
    planner_result_dir = DDPG_RESULT
  elif args.pi.find('unif')!=-1:
    planner_result_dir = UNIF_RESULT
  
  print "Loading pick weights",pick_weight_file
  policy.pick_pi.a_gen.load_weights(pick_weight_file)
  print "Loading place weights",place_weight_file
  policy.place_pi.a_gen.load_weights(place_weight_file)
  return planner_result_dir

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
  elif args.pi == 'gail':
    pick_dir = "/data/public/rw/pass.port/NAMO/n_data_"+str(n_data)+"/gail/adv/tau_0.3/dg_lr_0.0001_0.0001/n_score_1/architecture_0/"
    place_dir = "/data/public/rw/pass.port/NAMO/n_data_"+str(n_data)+"/gail/adv/tau_0.3/dg_lr_0.0001_0.0001/n_score_1/architecture_0/"
    eval_dir = pick_dir+'/explr_const_0.5/'
  else:
    print "Not set yet"
    sys.exit(-1)

  return pick_dir,place_dir,eval_dir
 
def load_weights( args,policy ):
  n_data  = args.n_data
  n_trial = args.n_trial
  epoch   = args.epoch

  pick_dir,place_dir,eval_dir = get_pick_place_eval_dirs(args)
  pick_wdir = pick_dir + '/n_trial_'+str(n_trial)+'/train_results/'
  place_wdir = place_dir + '/n_trial_'+str(n_trial)+'/train_results/'

  pick_weight = [wfile for wfile in os.listdir(pick_wdir) \
                  if (wfile.find('a_gen_pick_epoch_'+str(epoch)+'_')!=-1) \
                    or (wfile.find('a_gen_pick_epoch_'+str(epoch)+'.')!=-1)][0]
  place_weight = [wfile for wfile in os.listdir(place_wdir) \
                  if( wfile.find('a_gen_place_epoch_'+str(epoch)+'_')!=-1) or\
                    (wfile.find('a_gen_place_epoch_'+str(epoch)+'.')!=-1) ][0]
  print "pick loading weights",pick_wdir+pick_weight
  print "place loading weights",place_wdir+place_weight
  policy.pick_pi.a_gen.load_weights(pick_wdir+pick_weight)
  policy.place_pi.a_gen.load_weights(place_wdir+place_weight)
  
def set_scalers(args,policy):
  n_data  = args.n_data
  n_trial = args.n_trial

  pick_dir,place_dir,eval_dir = get_pick_place_eval_dirs(args)
  pick_scaler_dir = pick_dir+'/n_trial_'+str(n_trial)+'/scalers/'
  place_scaler_dir = place_dir+'/n_trial_'+str(n_trial)+'/scalers/'
  print "Setting pick place scalers"
  print pick_scaler_dir
  print place_scaler_dir

  policy.pick_pi.a_scaler     = pickle.load(open(pick_scaler_dir+'/pick_a_scaler.pkl','r'))
  policy.place_pi.a_scaler    = pickle.load(open(place_scaler_dir+'/place_a_scaler.pkl','r'))
  policy.pick_pi.misc_scaler  = pickle.load(open(pick_scaler_dir+'/pick_misc_scaler.pkl','r'))
  policy.place_pi.misc_scaler = pickle.load(open(place_scaler_dir+'/place_misc_scaler.pkl','r'))
  
def main():
  args      = parse_args()
  args.explr_const = 0.5

  problem = NAMO()
  if args.pi == 'unif':
    place_pi = UniformPlace( problem.problem['env'], \
                             problem.problem['obj_region'], \
                             problem.problem['all_region'] )
    pick_pi  = UniformPick( problem.problem['env'], \
                            problem.problem['obj_region'], \
                            problem.problem['all_region'] )
    planner_result_dir = UNIF_RESULT
    if not os.path.isdir(planner_result_dir):
      os.makedirs(planner_result_dir)
    nodes,rwd_time_list = forward_dfs_search(problem,pick_pi,place_pi,max_exp=50,visualize=args.v)
  else:
    policy = create_policy(args,problem.problem)
    load_weights( args,policy )
    set_scalers(args,policy)
    _,_,eval_dir = get_pick_place_eval_dirs(args)
    planner_result_dir = eval_dir+'/n_trial_'+str(args.n_trial)+'/planner_result/'
    if not os.path.isdir(planner_result_dir):
      os.makedirs(planner_result_dir)
    nodes,rwd_time_list = forward_dfs_search(problem,\
                                             policy.pick_pi,\
                                             policy.place_pi,\
                                             max_exp=50,\
                                             visualize=args.v)

  nodes_to_save = [n for n in nodes]
  for n in nodes_to_save: 
    n.state = n.state[1]

  pickle.dump({ 'nodes':nodes,\
                'rwd_time_list':rwd_time_list},\
                open(planner_result_dir+'result_'+str(args.pidx) +'.pkl','wb'))
  problem.problem['env'].Destroy()

if __name__ == '__main__':
  main()

