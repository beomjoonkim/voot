import sys
import numpy as np
import os
import cPickle as pickle
import time


sys.path.append('../mover_library/')
from samplers import *
from utils import *
from preprocessing_utils import setup_env

from BOX import BOX
import random
GRAB_SLEEP_TIME=0.05

def aggregate_data( processed_dir ): 
  plans = []
  env_idxs=[]
  # remember that I need to revive the environment
  for fin in os.listdir( processed_dir ):
    if fin.find('box') == -1: continue  
    env_idx = int(fin.split('box_place_train_data_')[1].split('.p')[0])
    plans.append ( pickle.load(open(processed_dir+fin,'r'))[0] )
    env_idxs.append( env_idx )

  pickle.dump( [plans,env_idxs],open('./processed_train_data/box_aggregated_data.py','wb'))

def try_plan(problem,plan):
  initial_state  = problem['initial_state']
  OBSTACLES      = problem['obstacles']
  OBJECTS        = problem['objects']
  loading_region = problem['loading_region']
  all_region     = problem['all_region']
  env            = problem['env']
  obj_shapes     = problem['obj_shapes']
  init_base_conf = problem['init_base_conf']
  problem['initial_state'][0].Restore()

  robot = env.GetRobots()[0]
  leftarm_manip = robot.GetManipulator('leftarm')
  rightarm_manip = robot.GetManipulator('rightarm')
  rightarm_torso_manip = robot.GetManipulator('rightarm_torso')

  g_time_list = []
  g_score_list = []


  curr_obj_idx=0  
  n_fails = 0
  max_packed = 0
  rwd_time_list = []
  #env.SetViewer('qtcoin')

  while curr_obj_idx < len(OBJECTS):
    curr_obj = OBJECTS[curr_obj_idx]
    curr_obj_shape = obj_shapes[curr_obj.GetName()]
    curr_obj_original_trans = curr_obj.GetTransform()

    # choose a pick
    pick_base_pose=None
    while pick_base_pose is None:
      """
      pick_base_pose,grasp_params,g_config = sample_pick( curr_obj,curr_obj_shape,\
                                                           robot,pick_gen,\
                                                           env,all_region )
      """
      print 'Sampling pick....'
      pick_base_pose,grasp_params,g_config = sample_pick( curr_obj,robot,env,all_region )
      if pick_base_pose is None:
        return None,None

      set_robot_config( pick_base_pose,robot )
      pick_obj( curr_obj,robot,g_config,leftarm_manip,rightarm_torso_manip )
      set_robot_config(init_base_conf,robot)
      print robot.GetGrabbed()
      if env.CheckCollision(curr_obj) or env.CheckCollision(robot):
        sleep(0.05)
        robot.Release(curr_obj)
        curr_obj.SetTransform(curr_obj_original_trans)
        continue
    p_samples = [plan[0][curr_obj_idx][:3]]
    place_obj_pose,place_robot_pose,path = sample_placement_using_gen(env,curr_obj,robot,p_samples,loading_region,all_region) 
    sleep(0.05)

    if place_robot_pose is not None:
      set_robot_config( place_robot_pose,robot)
      place_obj( curr_obj,robot,FOLDED_LEFT_ARM,leftarm_manip,rightarm_manip)
      set_robot_config(init_base_conf,robot) # NOTE: I am not planning back to the initial pose
      curr_obj_idx+=1
    else:
      return curr_obj_idx
  return curr_obj_idx

def main():
  np.random.seed(1) # important because BOX assumes same valule
  random.seed(1)
  
  n_data  = int(sys.argv[1])
  n_trial = int(sys.argv[2])
  raw_file_idx = int(sys.argv[3])

  score_file_name = './box/n_data_'+str(n_data)+'/n_trial_'+str(n_trial)+'/train_results/score_vec_'+str(raw_file_idx)+'.p'
  if os.path.isfile(score_file_name): 
    print 'already have'
    return
  parent_dir = './box/n_data_'+str(n_data)
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
  
  fdata = './processed_train_data/box_aggregated_data.py'
  if not os.path.exists( fdata ):
    plans = aggregate_data( './processed_train_data/' )
  plans = pickle.load(open( fdata,'r') )[0]
  import pdb;pdb.set_trace()
  
   
  raw_dir       = './train_data/'
  raw_file_list = os.listdir(raw_dir)
  raw_file = raw_file_list[raw_file_idx]
  
  search_episode_data = pickle.load(open(raw_dir+raw_file,'r'))
  problem = setup_env(search_episode_data)

  scores = []
  for plan in plans:
    score = try_plan(problem,plan)
    print score
    scores.append(score)
    pickle.dump(scores,open(score_file_name,'wb'))
    






if __name__ == '__main__':
  main()
