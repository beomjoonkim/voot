import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf 
import copy

from PlaceSOAP import PlaceSOAP
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

import warnings
import keras
import random

from manipulation.bodies.bodies import box_body, randomly_place_body, place_xyz_body
from manipulation.primitives.transforms import *


from preprocessing_utils import setup_env,aggregate_data,get_trajectories,get_node_idx
sys.path.append('../mover_library/')
from utils import set_robot_config,draw_configs
from manipulation.bodies.bodies import set_transparency,set_color
from matplotlib import pyplot as plt
from utils import compute_occ_vec
from data_load_utils import create_bit_encoding_of_konf,clean_pose_data
from data_load_utils import convert_collision_vec_to_one_hot

def draw_robot_at_conf( conf,transparency,name,robot,env ):
  newrobot = RaveCreateRobot(env,robot.GetXMLId())
  newrobot.Clone(robot,0)
  newrobot.SetName(name)
  env.Add(newrobot,True)
  set_robot_config( conf, newrobot ) 
  for link in newrobot.GetLinks():
    for geom in link.GetGeometries():
      geom.SetTransparency( transparency )

def predict_at_given_conf( conf,w_data,scaled_c,scaled_k,evaluator,robot):
  conf = conf[None,:]
  scaled_x = conf
  op= evaluator.disc.predict([scaled_x,w_data,scaled_c,scaled_k])
  H1=evaluator.H1_model.predict([scaled_x,w_data,scaled_c,scaled_k])
  H1 = H1.squeeze()
  return op,H1

def visualize_target_conf(target_conf,w_data,c_data,k_data,robot,key_configs,env,evaluator):
  for body in env.GetBodies():
    if body.GetName().find('conf')!=-1 or body.GetName().find('proposed')!=-1:
      env.Remove(body)
  
  # want to be here
  set_robot_config( target_conf,robot )
  x_scaler=None
  score,H1=predict_at_given_conf( target_conf,w_data,c_data,k_data,evaluator,robot)
  
  # visualize the objects
  curr_obj = robot.GetGrabbed()[0]
  robot.Release(curr_obj)
  maxH1 = np.max(H1,1)
  maxH1 /= np.max(maxH1)

  highest_idxs = np.argsort(maxH1)[::-1][0:5]
  highest_vals = maxH1[np.argsort(maxH1)[::-1][0:5]]
  for idx,hidx in enumerate(highest_idxs):
    print idx,hidx
    draw_robot_at_conf(key_configs[hidx],1-highest_vals[idx]*0.5 ,'conf'+str(hidx),robot,env)
  robot.Grab(curr_obj)

  return score,highest_vals,highest_idxs

def pick_obj(obj,robot,g_configs,left_manip,right_manip):
  GRAB_SLEEP_TIME=0.05
  set_config(robot, g_configs[0],left_manip.GetArmIndices())
  set_config(robot, g_configs[1],right_manip.GetArmIndices())
  sleep(GRAB_SLEEP_TIME)
  robot.Grab(obj)


def main():
  n_data = int(sys.argv[1])
  n_trial = sys.argv[2]

  # select a raw environment file
  raw_dir       = './train_data/'
  raw_file_idx  = 12
  raw_file_list = os.listdir(raw_dir)
  raw_file      = raw_file_list[raw_file_idx]
  env_num       = raw_file.split('.pkl')[0]
  env_file = pickle.load(open(raw_dir+raw_file,'r'))
  
  # setup an environment
  problem = setup_env(env_file)
  env = problem['env']
  robot = env.GetRobots()[0]
  leftarm_manip = robot.GetManipulator('leftarm')
  rightarm_manip = robot.GetManipulator('rightarm')
  rightarm_torso_manip = robot.GetManipulator('rightarm_torso')

  # pick an object
  for node in env_file['nodes']:
    if node.sample is not None and 'pick_base_pose' in node.sample.keys():
      break
  obj_name  = node.state[0]
  curr_obj  = env.GetKinBody(obj_name)
  g_config  = node.sample['g_config']

  # dir setup
  parent_dir = './place_soap/n_data_'+str(n_data)
  trial_dir = parent_dir + '/n_trial_' + str(n_trial)
  scaler_dir = trial_dir + '/scalers/'
  train_results_dir = trial_dir + '/train_results/'

  x_scaler    = pickle.load(open(scaler_dir+'/x_scaler.p','r')) #NOTE: not used
  c0_scaler   = pickle.load(open(scaler_dir+'/c0_scaler.p','r')) #NOTE not used
  o_scaler = pickle.load(open(scaler_dir+'/tobj_scaler.p','r')) #NOTE not used

  # load konfs
  key_configs = np.array(pickle.load( open('./place_key_configs.p','r')))
  key_config_idxs = pickle.load(  open('./place_key_config_idxs_n_'+str(n_data)+'.p','r'))
  key_configs = key_configs[key_config_idxs,:]
  n_key_confs = key_configs.shape[0]

  env.SetViewer('qtcoin')

  # draw initialize location
  base_pose = np.array(node.sample['pick_base_pose'])
  set_robot_config(base_pose,robot)
  pick_obj( curr_obj,robot,g_config,leftarm_manip,rightarm_torso_manip )

  # compute c and k and scale them; only C scales atm
  c_data = compute_occ_vec(key_configs,robot,env)[None,:]*1
  c_data = convert_collision_vec_to_one_hot(c_data)

  #draw_robot_at_conf( base_pose,0.7,'init',robot,env)
  set_robot_config(base_pose,robot)

  # create w
  tobj_pose = np.array(problem['obj_poses']['obj0'])[None,:]
  base_pose = base_pose[None,:]
  clean_pose_data(tobj_pose)
  clean_pose_data(base_pose)
  c0_data = c0_scaler.transform(base_pose)
  o_data = o_scaler.transform(tobj_pose)
  w_data = np.hstack([base_pose,o_data])
  w_data = c0_data
  

  session = tf.Session()
  dim_x   = 3                        # data shape
  dim_w   = 3                        # context vector shape
  dim_c   = (n_key_confs,2)                    # collision vector shape
  dim_k   = (n_key_confs,12,1) # konf data shape


  # load the place soap
  wfile='a_genepoch_230_3000_Dtrue_0.513438_Dfake_-6.16826.h5' 
  wfile='a_genepoch_70_3000_Dtrue_-0.21714_Dfake_-0.885222.h5'
  wfile='a_genepoch_170_3000_Dtrue_0.223133_Dfake_-4.07246.h5'
  wfile='a_genepoch_150_3000_Dtrue_-0.125322_Dfake_-6.34039.h5'
  soap = PlaceSOAP(session,dim_x,dim_w,dim_k,dim_c,\
                   save_folder=train_results_dir)  
  soap.a_gen.load_weights( train_results_dir+wfile)
  
  
  n_gen = 200
  zvals = np.random.normal(size=(n_gen,soap.dim_z)).astype('float32')
  w_data = np.tile(w_data,(n_gen,1))
  c_data = np.tile(c_data,(n_gen,1,1))
  Gpred = soap.a_gen.predict( [zvals,w_data,c_data] )
  Gpred = x_scaler.inverse_transform(Gpred)
  draw_configs(configs=Gpred,env=env,name='conf',colors=(1,0,0),transparency=0)
  import pdb;pdb.set_trace()

  for p in Gpred:
    set_robot_config(p,robot)
    import pdb;pdb.set_trace()
if __name__ == '__main__':
  main()





