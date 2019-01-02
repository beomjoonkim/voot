import sys
import os
import pickle
import time
import tensorflow as tf
import numpy as np

sys.path.append('../')
sys.path.append('../../mover_library/')
from SOAP import SOAP
from PlaceSOAP import PlaceSOAP
from conveyor_belt_problem import two_tables_through_door
from openravepy import *


from samplers import  *
from mover_problem import sample_pick,sample_placement,pick_obj,place_obj
from operator_utils.grasp_utils import solveTwoArmIKs
from operator_utils.grasp_utils import compute_two_arm_grasp
from misc.priority_queue import Stack, Queue, FILOPriorityQueue, PriorityQueue
from TreeNode import *

from manipulation.primitives.transforms import get_point
from manipulation.primitives.savers import DynamicEnvironmentStateSaver
from manipulation.constants import FOLDED_LEFT_ARM

def compute_occ_vec(key_configs,robot,env):
  occ_vec = []
  for config in key_configs:
    set_robot_config(config,robot)
    occ_vec.append( env.CheckCollision(robot) )
  return np.array(occ_vec)

def set_obj_config(xytheta,obj):
  x = xytheta[0]
  y = xytheta[1]
  z = get_point(obj)[-1]
  set_point(obj,[x,y,z])
  th= xytheta[2]
  set_quat(obj,quat_from_angle_vector(th,np.array([0,0,1])))

def test_place_generator_time(problem,place_gen,pick_gen,key_configs):
  initial_state  = problem['initial_state']
  OBSTACLES      = problem['obstacles']
  OBJECTS        = problem['objects']
  loading_region = problem['loading_region']
  all_region     = problem['all_region']
  env            = problem['env']
  obj_shapes     = problem['obj_shapes']
  init_base_conf = problem['init_base_conf']
  #draw_configs(env,key_configs,'key',(1,1,1))

  robot = env.GetRobots()[0]
  leftarm_manip = robot.GetManipulator('leftarm')
  rightarm_manip = robot.GetManipulator('rightarm')
  rightarm_torso_manip = robot.GetManipulator('rightarm_torso')

  g_time_list = []
  g_score_list = []
  stime=time.time()

  n_fails=0

  curr_obj_idx=0  
  n_fails = 0
  max_packed = 0
  rwd_time_list = []
  while curr_obj_idx < len(OBJECTS) and n_fails<5:
    curr_obj = OBJECTS[curr_obj_idx]
    curr_obj_shape = obj_shapes[curr_obj.GetName()]
    curr_obj_original_trans = curr_obj.GetTransform()

    # choose a pick
    pick_base_pose=None
    while pick_base_pose is None:
      pick_base_pose,grasp_params,g_config = sample_pick_using_gen( curr_obj,curr_obj_shape,\
                                                                    robot,pick_gen,\
                                                                    env,all_region )
      # set robot to its init pose
      if pick_base_pose is None:
        continue
    set_robot_config(pick_base_pose,robot)
    pick_obj( curr_obj,robot,g_config,leftarm_manip,rightarm_torso_manip )

    
    # compute occ_vec
    print 'computing occ vec'
    occ_vec = compute_occ_vec(key_configs,robot,env)
    print 'done'
    set_robot_config(init_base_conf,robot)  

    # draw obj placement distribution
    #p_place = [generate_obj_placement(place_gen,occ_vec)[0] for _ in range(100)] 
    #color=1./(curr_obj_idx+1)
    #draw_configs(env,p_place,str(time.time()),(1,0,color))

    place_obj_pose,place_robot_pose,path = sample_placement_using_gen(env,curr_obj,\
                                                                      robot,place_gen,\
                                                                      occ_vec,\
                                                                      loading_region,\
                                                                      all_region)
    sleep(0.05)
    robot.Release(curr_obj)
    if place_obj_pose is not None:
      curr_obj_idx+=1
    else:
      problem['initial_state'][0].Restore()
      curr_obj_idx=0
      n_fails+=1
    rwd_time_list.append([curr_obj_idx,time.time()-stime])
  return rwd_time_list
  
  """
  import pdb;pdb.set_trace()

  obst = problem['obstacles'][0]
  set_obj_config(p_place[0],obst)


  occ_vec2 = compute_occ_vec(key_configs,robot,env)
  set_robot_config(init_base_conf,robot)  
  p_place = [generate_obj_placement(place_gen,occ_vec2)[0] for _ in range(100)] 
  draw_configs(env,p_place,str(time.time()),(1,0,0))
  import pdb;pdb.set_trace()
  """



  # pick new object

  # compute new occ_vec

  # check the object placement distribution
  """
  while curr_obj_idx < len(OBJECTS) and n_fails<100: # you can fail at most 5 times
    curr_obj = OBJECTS[curr_obj_idx]
    curr_obj_shape = obj_shapes[curr_obj.GetName()]
    curr_obj_original_trans = curr_obj.GetTransform()
    pick_base_pose=None
    while pick_base_pose is None:
      pick_base_pose,grasp_params,g_config = sample_pick_using_gen( curr_obj,curr_obj_shape,\
                                                                    robot,pick_gen,\
                                                                    env,all_region )
      # set robot to its init pose
      if pick_base_pose is None:
        continue
      set_robot_config(pick_base_pose,robot)
      pick_obj( curr_obj,robot,g_config,leftarm_manip,rightarm_torso_manip )
    stime=time.time()
    occ_vec = compute_occ_vec(key_configs,robot,env)
    set_robot_config(init_base_conf,robot)  
    place_obj_pose,place_robot_pose,path = sample_placement_using_gen(env,curr_obj,\
                                                                      robot,place_gen,\
                                                                      occ_vec,\
                                                                      loading_region,\
                                                                      all_region)
    if place_obj_pose is None:
      # should I reset?
      n_fails+=1
      sleep(0.1)
      robot.Release(curr_obj)
      curr_obj.SetTransform(curr_obj_original_trans)
      set_robot_config(init_base_conf,robot)
    else:
      set_robot_config(place_robot_pose,robot)
      place_obj( curr_obj,robot,FOLDED_LEFT_ARM,leftarm_manip,rightarm_manip)
      curr_obj_idx+=1
  """
  return curr_obj_idx,time.time()-stime
      

def define_generator(weight_f,n_data,n_trial):
  parent_dir = '../place_soap/n_data_'+str(n_data)
  trial_dir = parent_dir + '/n_trial_' + str(n_trial)
  scaler_dir = trial_dir + '/scalers/'
  train_results_dir = trial_dir + '/train_results/'
  test_results_dir = trial_dir +'/test_results/'
  if not os.path.exists(test_results_dir):
    os.mkdir(test_results_dir)

  sess        = tf.Session()
  dim_data    = 3
  dim_context = 1620

  x_scaler = pickle.load( open(scaler_dir+'x_scaler.p','r') )
  c_scaler = pickle.load( open(scaler_dir+'c_scaler.p','r') )
  generator = PlaceSOAP(sess,dim_data,dim_context,train_results_dir,x_scaler,c_scaler)
  generator.load_offline_weights(weight_f)
  generator.generate(np.zeros((1,dim_context,1)),1) # compilation purpose
  return generator

def define_pick_generator():
  n_data  = 11000
  n_trial = 0
  weight_f  = 'a_genepoch_530_3000_Dtrue_1.79797_Dfake_-5.83497.h5'
  parent_dir = '../soap/n_data_'+str(n_data)
  trial_dir = parent_dir + '/n_trial_' + str(n_trial)
  scaler_dir = trial_dir + '/scalers/'
  train_results_dir = trial_dir + '/train_results/'

  sess        = tf.Session()
  dim_data    = 6
  dim_context = 3

  x_scaler = pickle.load( open(scaler_dir+'x_scaler.p','r') )
  c_scaler = pickle.load( open(scaler_dir+'c_scaler.p','r') )
  generator = SOAP(sess,dim_data,dim_context,train_results_dir,x_scaler,c_scaler)
  generator.load_offline_weights(weight_f)
  generator.generate(np.array([[0,0,0]]),1) # compilation purpose
  return generator


min_height = 0.4
max_height = 1

min_width = 0.2
max_width = 0.6

min_length = 0.2
max_length = 0.6

def create_custom_env():
  env=Environment()
  #env.SetViewer('qtcoin')
  problem = two_tables_through_door(env)
  return problem

def draw_configs_with_color(env,configs,name,colors):
  # helper function for debugging
  for i in range(len(configs)):
    config = configs[i] 
    color=colors[i]
    new_body = box_body(env,0.05,0.05,0.05,\
                        name=name+'_pt_%d'%i,\
                        color=colors[i])
    env.Add(new_body); set_point(new_body,np.append(config[0:2],0.075))
    new_body.Enable(False)

def main():
  n_data = 5000
  n_trial = 0

  # directory setup
  parent_dir = '../place_soap/n_data_'+str(n_data)
  trial_dir = parent_dir + '/n_trial_' + str(n_trial)
  scaler_dir = trial_dir + '/scalers/'
  train_results_dir = trial_dir + '/train_results/'

  if not os.path.exists(parent_dir):
    os.mkdir(parent_dir)
  if not os.path.exists(trial_dir):
    os.mkdir(trial_dir)
  if not os.path.exists(scaler_dir):
    os.mkdir(scaler_dir)
  if not os.path.exists(train_results_dir):
    os.mkdir(train_results_dir)

  proc_train_data_dir = '../processed_train_data/'
  data = pickle.load( open( proc_train_data_dir+'place_aggregated_data.p','r') )
  n_data = min(n_data,len(data[0]))

  x_data = np.array(data[0])[:n_data,:3] # actions - predict object pose
  s_data = np.array(data[1])[:n_data,:]  # scores
  oidx_data = data[3][:n_data,:]



  env=Environment()
  problem = two_tables_through_door(env)
  env.SetViewer('qtcoin')
  
  score_list=range(1,6)
  oidx_list = range(0,5)

  score_list=[1]
  oidx_list=[0]
  pairs =[ [s,o] for s in score_list for o in oidx_list]
  for p in pairs:
    print p
    s=p[0]
    o=p[1]
    target_idx = (s_data==s) * (oidx_data==o)
    print np.sum(target_idx)
    color=(o/float(4),0,0)
    draw_configs(x_data[target_idx[:,0]],env,name=str(s)+str(o),colors=color)#transparency=(4-o)/float(5))
  import pdb;pdb.set_trace()

    
  
if __name__ == '__main__':
  main()
