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


from preprocessing_utils import setup_env,get_trajectories,get_node_idx
sys.path.append('../mover_library/')
from utils import set_robot_config,draw_configs,compute_occ_vec

from manipulation.bodies.bodies import set_transparency,set_color

from matplotlib import pyplot as plt
from data_load_utils import load_place_data,convert_collision_vec_to_one_hot


def pick_obj(obj,robot,g_configs,left_manip,right_manip):
  GRAB_SLEEP_TIME=0.05
  set_config(robot, g_configs[0],left_manip.GetArmIndices())
  set_config(robot, g_configs[1],right_manip.GetArmIndices())
  sleep(GRAB_SLEEP_TIME)
  robot.Grab(obj)


def predict_at_given_conf( scaled_x,scaled_c,scaled_k,evaluator,robot):
  op= evaluator.disc.predict([scaled_x,scaled_k,scaled_c])
  H1 = evaluator.H1_model.predict([scaled_x,scaled_k,scaled_c])
  H1 = H1.squeeze()
  return op,H1

def visualize_target_conf(scaled_x,scaled_c,scaled_k,key_configs,env,robot,evaluator):
  for body in env.GetBodies():
    if body.GetName().find('conf')!=-1 or body.GetName().find('proposed')!=-1:
      env.Remove(body)
  
  score,H1=predict_at_given_conf( scaled_x,scaled_c,scaled_k,evaluator,robot)
  
  curr_obj = robot.GetGrabbed()[0]
  # visualize the objects
  robot.Release(curr_obj)
  maxH1 = np.max(H1,1)
  maxH1 /= np.max(maxH1)

  highest_idxs = np.argsort(maxH1)[::-1][0:10]
  highest_vals = maxH1[np.argsort(maxH1)[::-1][0:10]]
  for idx,hidx in enumerate(highest_idxs):
    newrobot = RaveCreateRobot(env,robot.GetXMLId())
    newrobot.Clone(robot,0)
    newrobot.SetName('conf'+str(hidx))
    env.Add(newrobot,True)
    set_robot_config(key_configs[hidx], newrobot) 
    for link in newrobot.GetLinks():
      for geom in link.GetGeometries():
        geom.SetTransparency( 1-highest_vals[idx]*0.5  )
  robot.Grab(curr_obj)

  return score,highest_vals,highest_idxs



# make an OpenRAVE conveyor belt problem
#raw_file_idx = int(sys.argv[1])

def main():
  raw_dir       = './train_data/'
  raw_file_idx  = 45
  raw_file_list = os.listdir(raw_dir)
  raw_file      = raw_file_list[raw_file_idx]
  env_num       = raw_file.split('.pkl')[0]
  env_file = pickle.load(open(raw_dir+raw_file,'r'))
  problem = setup_env(env_file)
  env = problem['env']
  robot = env.GetRobots()[0]
  leftarm_manip = robot.GetManipulator('leftarm')
  rightarm_manip = robot.GetManipulator('rightarm')
  rightarm_torso_manip = robot.GetManipulator('rightarm_torso')

  # init location
  """
  newrobot = RaveCreateRobot(env,robot.GetXMLId())
  newrobot.Clone(robot,0)
  newrobot.SetName('init')
  env.Add(newrobot,True)
  set_robot_config(problem['init_base_conf'], newrobot) 
  for link in newrobot.GetLinks():
    for geom in link.GetGeometries():
      geom.SetTransparency( 0 )
  """

  # pick an object
  node      = env_file['nodes'][1]
  obj_name  = node.sample['obj'] 
  curr_obj  = env.GetKinBody(obj_name)
  base_pose = node.sample['pick_base_pose']
  g_config  = node.sample['g_config']
  set_robot_config(base_pose,robot)
  pick_obj( curr_obj,robot,g_config,leftarm_manip,rightarm_torso_manip )
  set_robot_config(problem['init_base_conf'],robot)

  
  # directory setup
  n_data = int(sys.argv[1])
  n_trial = sys.argv[2]
  proc_train_data_dir = 'processed_train_data/'
  data = pickle.load( open( proc_train_data_dir+'place_aggregated_data.p','r') )
  n_data = min(n_data,len(data[0]))

  parent_dir = './place_evaluator/n_data_'+str(n_data)
  trial_dir = parent_dir + '/n_trial_' + str(n_trial)
  scaler_dir = trial_dir + '/scalers/'
  train_results_dir = trial_dir + '/train_results/'

  # get key configs
  key_configs = np.array(pickle.load( open('./key_configs/key_configs.p','r')))
  key_config_idxs = pickle.load(  open('./key_configs/key_config_idxs_n_'+str(n_data)+'.p','r'))
  key_configs = key_configs[key_config_idxs,:]
  n_key_confs = key_configs.shape[0]

  # compute collision vector
  c_data = compute_occ_vec(key_configs,robot,env)[None,:]
  #c_data = c_data.reshape((1,1*n_key_confs))
  #c_scaler = pickle.load( open(scaler_dir+'/c_scaler.p','r'))
  #scaled_c = c_scaler.transform( c_data )
  scaled_c = convert_collision_vec_to_one_hot(c_data)

  dim_data    = 3
  dim_context = (n_key_confs,scaled_c.shape[2]) # single channel
  dim_konf = (n_key_confs,3,1)

  # dummy k
  """
  k_data = key_configs
  k_scaler = pickle.load( open(scaler_dir+'/k_scaler.p','r'))
  k_data = k_data.reshape((1,3*n_key_confs))
  scaled_k = k_scaler.transform( k_data )
  scaled_k = scaled_k.reshape((1,n_key_confs,3,1))
  """
  data = load_place_data( parent_dir='./place_soap/', \
                          proc_train_data_dir='processed_train_data',\
                          n_data=n_data,n_trial=n_trial)
  scaled_k = data['k']
  session = tf.Session()
  evaluator = PlaceEvaluator(session,dim_data,dim_context,dim_konf,train_results_dir)
  evaluator.disc.load_weights(train_results_dir+'weights.05.hdf5')

  x_scaler = pickle.load( open(scaler_dir+'/x_scaler.p','r'))
  #scaled_x = x_scaler.transform( x_data ) 
  #score,H1,highest_idxs = visualize_target_conf(scaled_x,scaled_c,scaled_k,key_configs,env,robot,evaluator );
  T_r_wrt_o = np.dot( np.linalg.inv( curr_obj.GetTransform()), robot.GetTransform())
  def visualize(obj_xytheta):
    obj_xytheta=obj_xytheta.squeeze()
    robot.Release(curr_obj)
    objx = obj_xytheta[0]
    objy = obj_xytheta[1]
    objz = get_point(curr_obj)[-1]
    set_point(curr_obj,[objx,objy,objz])
    th=obj_xytheta[2]
    set_quat( curr_obj, quat_from_z_rot(th) )
    new_T_robot = np.dot( curr_obj.GetTransform(),T_r_wrt_o) 
    robot.SetTransform(new_T_robot)
    robot.Grab(curr_obj)

    scaled_x = x_scaler.transform( obj_xytheta[None,:] ) 
    score,H1,highest_idxs = visualize_target_conf(scaled_x,scaled_c,scaled_k,key_configs,env,robot,evaluator );
    print score
  env.SetViewer('qtcoin')
  visualize( np.array([[-1.5,0.5,270*np.pi/180]]))
  import pdb;pdb.set_trace()
  visualize( np.array([[-1.5,0,0*np.pi/180]]))
    

  import pdb;pdb.set_trace()

   

if __name__ == '__main__':
  main()

