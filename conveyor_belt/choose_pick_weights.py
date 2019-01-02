import sys
import os
import pickle
import time
import tensorflow as tf
import numpy as np
from SOAP import SOAP
from conveyor_belt_problem import two_tables_through_door
from openravepy import *


sys.path.append('../mover_library/')
from samplers import  *
from mover_problem import sample_pick,sample_placement,pick_obj,place_obj
from operator_utils.grasp_utils import solveTwoArmIKs
from operator_utils.grasp_utils import compute_two_arm_grasp
from misc.priority_queue import Stack, Queue, FILOPriorityQueue, PriorityQueue
from TreeNode import *

from manipulation.primitives.transforms import get_point
from manipulation.primitives.savers import DynamicEnvironmentStateSaver
from manipulation.constants import FOLDED_LEFT_ARM


def draw_configs(env,configs,name,color):
  for i in range(len(configs)):
    config = configs[i]
    new_body = box_body(env,0.05,0.05,0.05,\
                        name=name+'_pt_%d'%i,\
                        color=color)
    env.Add(new_body); set_point(new_body,np.append(config[0:2],0.075))

def visualize_base_poses( problem, generator ):
  # setup the environment
  OBSTACLES      = problem['obstacles']
  OBJECTS        = problem['objects']
  loading_region = problem['loading_region']
  all_region     = problem['all_region']
  env            = problem['env']
  obj_shapes     = problem['obj_shapes']
  
  robot = env.GetRobots()[0]
  for idx,curr_obj in enumerate(OBJECTS):
    curr_obj_shape = obj_shapes[curr_obj.GetName()]
    base_xy = []
    for _ in range(100):
      theta,height_portion,depth_portion,base_pose \
          = generate_pick_grasp_and_base_pose(generator,curr_obj_shape,get_point(curr_obj))
      base_xy.append(base_pose[0:2])
    draw_configs(env,base_xy,curr_obj.GetName(),color=(0,1.0/(len(OBJECTS)-(idx)),0))

def test_generator_time(problem,generator=None):
  # setup the environment
  initial_state  = problem['initial_state']
  OBSTACLES      = problem['obstacles']
  OBJECTS        = problem['objects']
  loading_region = problem['loading_region']
  all_region     = problem['all_region']
  env            = problem['env']
  obj_shapes     = problem['obj_shapes']
  
  robot = env.GetRobots()[0]

  # get object to grasp
  g_time_list = []
  g_score_list = []
  for curr_obj in OBJECTS:
    curr_obj_shape = obj_shapes[curr_obj.GetName()]
    g_time  = [] 
    g_solns = 0
    for _ in range(1):
      stime=time.time()
      if generator is None:
        pick_base_pose,grasp_params,g_config = sample_pick( curr_obj,\
                                                            robot,\
                                                            env,\
                                                            all_region )
      else:
        pick_base_pose,grasp_params,g_config = sample_pick_using_gen( curr_obj,curr_obj_shape,\
                                                                      robot,generator,\
                                                                      env,all_region )
      g_time.append(time.time()-stime)
      
      if g_config is not None: 
        g_solns+=1    
        print g_solns
      print np.mean(g_time)
    if np.mean(g_time) > 3.0 and generator is not None:
      # if any one of objs is giving trouble, skip this
      return None,None
    g_time_list.append(g_time)
    g_score_list.append(g_solns)
  return g_time_list,g_score_list

def define_generator(weight_f,n_data,n_trial):
  parent_dir = './soap/n_data_'+str(n_data)
  trial_dir = parent_dir + '/n_trial_' + str(n_trial)
  scaler_dir = trial_dir + '/scalers/'
  train_results_dir = trial_dir + '/train_results/'
  test_results_dir = trial_dir +'/test_results/'
  if not os.path.exists(test_results_dir):
    os.mkdir(test_results_dir)

  sess        = tf.Session()
  dim_data    = 6
  dim_context = 3

  x_scaler = pickle.load( open(scaler_dir+'x_scaler.p','r') )
  c_scaler = pickle.load( open(scaler_dir+'c_scaler.p','r') )
  generator = SOAP(sess,dim_data,dim_context,train_results_dir,x_scaler,c_scaler)
  generator.load_offline_weights(weight_f)
  generator.generate(np.array([[0,0,0]]),1) # compilation purpose
  
  print weight_f
  return generator

min_height = 0.4
max_height = 1

min_width = 0.2
max_width = 0.6

min_length = 0.2
max_length = 0.6

def main():
  n_data  = int(sys.argv[1])
  n_trial = int(sys.argv[2])

  env=Environment()
  #env.SetViewer('qtosg')

  height_vals = np.linspace(0.4,1,7) 
  width_vals  = np.linspace(0.2,0.6,5)
  length_vals = np.linspace(0.2,0.6,5)
  shape_list = [[w,l,h] for h in height_vals for w in width_vals for l in length_vals]
  
  obj_shapes = {}
  obj_poses  = {}
  for i,obj_shape in enumerate(shape_list):
    width  = obj_shape[0]
    length = obj_shape[1]
    height = obj_shape[2]
    obj_name = 'obj%s'%i
    obj_shapes[obj_name] = [width,length,height]
    obj_poses[obj_name]  = [3,3,3]
  print len(obj_shapes)
  problem = two_tables_through_door(env,obj_shapes,obj_poses,obst_shapes=None,obst_poses=None)
  
  train_results_dir = './soap/n_data_'+str(n_data) + '/n_trial_' \
                      + str(n_trial) + '/train_results/'
  performance_list = []

  TESTRAWSAMPLER = False
  if TESTRAWSAMPLER:
    g_time_list,g_score_list = test_generator_time(problem)
    performance_list.append( [g_time_list,g_score_list] )
    g_time_list = np.array(g_time_list);  g_score_list = np.array(g_score_list)
    print np.mean(g_time_list),np.mean(g_score_list)
    pickle.dump( performance_list, open('uniform_pick_performance.pkl','wb') )
  else:
    weight_f_list = os.listdir(train_results_dir)
    np.random.shuffle( weight_f_list )  
    for weight_f in weight_f_list:
      if weight_f.find('a_gen') == -1: continue
      #weight_f = 'a_genepoch_2580_3000_Dtrue_0.91743_Dfake_-1.60897.h5'
      weight_f = 'a_genepoch_470_3000_Dtrue_1.0384_Dfake_-13.6155.h5'
      generator   = define_generator(weight_f,n_data,n_trial)
      print weight_f
      g_time_list,g_score_list = test_generator_time(problem,generator)
      
      if g_time_list is None:
        continue
      print weight_f
      performance_list.append( [weight_f,g_time_list,g_score_list] )
      pickle.dump( performance_list, open('soap_pick_performance.pkl','wb') )


if __name__ == '__main__':
  main()
