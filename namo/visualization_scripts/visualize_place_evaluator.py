import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf 
import copy

from PlaceEvaluator import PlaceEvaluator
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
from data_load_utils import create_bit_encoding_of_konf,clean_pose_data,convert_collision_vec_to_one_hot,load_place_data

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
  scaled_x = conf
  op= evaluator.disc.predict([scaled_x,w_data,scaled_c])
  H1=evaluator.H1_model.predict([scaled_x,w_data,scaled_c])
  H1 = H1.squeeze()
  return op,H1

def visualize_target_conf(target_conf,w_data,c_data,k_data,robot,key_configs,env,evaluator,x_scaler):
  for body in env.GetBodies():
    if body.GetName().find('conf')!=-1 or body.GetName().find('proposed')!=-1:
      env.Remove(body)
  
  # want to be here
  set_robot_config( target_conf,robot )
  target_conf = x_scaler.transform(target_conf[None,:]) # transform data
  print target_conf
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
  n_trial = int(sys.argv[2])

  # select a raw environment file
  raw_dir       = './train_data/'
  raw_file_idx  = 10
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
  base_pose = node.sample['pick_base_pose']
  g_config  = node.sample['g_config']
  set_robot_config(base_pose,robot)
  pick_obj( curr_obj,robot,g_config,leftarm_manip,rightarm_torso_manip )


  # dir setup
  parent_dir = './place_evaluator/n_data_'+str(n_data)
  trial_dir = parent_dir + '/n_trial_' + str(n_trial)
  scaler_dir = trial_dir + '/scalers/'
  train_results_dir = trial_dir + '/train_results/'

  x_scaler    = pickle.load(open(scaler_dir+'/x_scaler.p','r')) #NOTE: not used
  c0_scaler   = pickle.load(open(scaler_dir+'/c0_scaler.p','r')) #NOTE not used
  o_scaler = pickle.load(open(scaler_dir+'/tobj_scaler.p','r')) #NOTE not used
  c_scaler = pickle.load(open(scaler_dir+'/c_scaler.p','r')) #NOTE not used
  k_scaler = pickle.load(open(scaler_dir+'/k_scaler.p','r')) #NOTE not used

  # load konfs
  key_configs = np.array(pickle.load( open('./place_key_configs.p','r')))
  key_config_idxs = pickle.load(  open('./place_key_config_idxs_n_'+str(n_data)+'.p','r'))
  key_configs = key_configs[key_config_idxs,:]
  n_key_confs = key_configs.shape[0]
  dim_konf = key_configs.shape[1]

  # compute c and k and scale them; only C scales atm
  c_data = compute_occ_vec(key_configs,robot,env)[None,:]*1
  c_data = convert_collision_vec_to_one_hot(c_data)
  #k_data = np.reshape( key_configs, (1,n_key_confs,3,1))

  # compute  k
  k_data = key_configs
  k_data = k_data.reshape((1,dim_konf*n_key_confs))
  k_data = k_scaler.transform( k_data )
  k_data = k_data.reshape((1,n_key_confs,dim_konf,1))
  
  # compute w
  set_robot_config(base_pose,robot)

  o_data = problem['obj_poses']['obj0'] 
  clean_pose_data(np.array(o_data)[None,:])
  o_data = o_scaler.transform(np.array(o_data)[None,:])

  clean_pose_data(np.array(base_pose)[None,:])
  c0_data = c0_scaler.transform(np.array(base_pose)[None,:])

  w_data = np.hstack([c0_data,o_data])
  w_data = c0_data


  session = tf.Session()
  dim_x   = 3                        # data shape
  dim_w   = 3                         # context vector shape
  dim_c   = (n_key_confs,2)                    # collision vector shape
  dim_k   = (n_key_confs,dim_konf,1) # konf data shape

  evaluator = PlaceEvaluator(session,dim_x,dim_w,dim_k,dim_c) 
  evaluator.disc.load_weights( train_results_dir+'weights.08.hdf5')

  from functools import partial
  # draw initialize location
  draw_robot_at_conf( base_pose,0,'init',robot,env)
  f = partial(visualize_target_conf,w_data=w_data,c_data=c_data,\
              k_data=k_data,robot=robot,key_configs=key_configs,env=env,evaluator=evaluator,x_scaler=x_scaler)

  #print f(np.array([-1.6,-1,90*np.pi/180]))
  #print f(np.array([-1.5,-1.5,0*np.pi/180]))
  #print f(np.array([-1,1.5,0*np.pi/180]))

  c_data = c_scaler.transform(c_data)
  scaled_x = x_scaler.transform( np.array([-1,1.5,0*np.pi/180])[None,:] )
  evaluator.disc.predict( [scaled_x,w_data,c_data] )
  import pdb;pdb.set_trace()

  """
  data = load_place_data( parent_dir='./place_evaluator/',\
                        proc_train_data_dir='./processed_train_data',\
                        n_data=n_data,n_trial=n_trial)
  x_data = data['x']
  c_data = data['c']
  k_data = data['k']
  s_data = data['s']*1
  c0_data = data['c0']
  o_data = data['o'] 
  """
 
  env.SetViewer('qtcoin')
  import pdb;pdb.set_trace()
if __name__ == '__main__':
  main()





