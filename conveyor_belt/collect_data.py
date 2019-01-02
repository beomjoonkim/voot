from manipulation.problems.fixed import ENVIRONMENTS_DIR
from manipulation.bodies.bodies import box_body, randomly_place_body, place_xyz_body
from manipulation.problems.problem import *
from manipulation.bodies.bodies import get_name
from misc.functions import randomize
from misc.generators import take
from misc.numerical import INF
from manipulation.bodies.robot import set_default_robot_config
from manipulation.primitives.transforms import get_point, set_point, pose_from_quat_point, unit_quat
from misc.colors import get_color
from manipulation.constants import BODY_PLACEMENT_Z_OFFSET
from manipulation.constants import *
from manipulation.primitives.utils import Pose

##TODO: Clean this
from manipulation.constants import PARALLEL_LEFT_ARM, REST_LEFT_ARM, HOLDING_LEFT_ARM, FOLDED_LEFT_ARM, FAR_HOLDING_LEFT_ARM, LOWER_TOP_HOLDING_LEFT_ARM,REGION_Z_OFFSET
from manipulation.regions import create_region, AARegion
from manipulation.bodies.bodies import randomly_place_region, place_body, place_body_on_floor
from manipulation.inverse_reachability.inverse_reachability import ir_base_trans
from manipulation.primitives.utils import mirror_arm_config
from manipulation.primitives.transforms import trans_from_base_values, set_pose, set_quat, \
  point_from_pose, axis_angle_from_rot, rot_from_quat, quat_from_pose, quat_from_z_rot,\
  get_pose,base_values_from_pose,pose_from_base_values, set_xy,quat_from_angle_vector,\
  quat_from_trans

from manipulation.primitives.savers import DynamicEnvironmentStateSaver

from itertools import product
import numpy as np
import copy
import math
import time 

from manipulation.bodies.bounding_volumes import aabb_extrema, aabb_from_body, aabb_union
from manipulation.inverse_reachability.inverse_reachability import get_custom_ir,\
                                                                   get_base_generator
from manipulation.bodies.robot import manip_from_pose_grasp
from manipulation.bodies.robot import get_active_arm_indices
from manipulation.grasps.grasps import FILENAME as GRASP_FILENAME, load_grasp_database
from manipulation.grasps.grasp_options import positive_hash, get_grasp_options
from manipulation.constants import GRASP_APPROACHES, GRASP_TYPES

from manipulation.bodies.bodies import geometry_hash
from manipulation.bodies.bodies import set_config
from manipulation.bodies.bounding_volumes import aabb_from_body
from manipulation.grasps.grasps import save_grasp_database, Grasp
from openravepy import *

# search episode
from misc.priority_queue import Stack, Queue, FILOPriorityQueue, PriorityQueue
from TreeNode import *


from manipulation.motion_planners.rrt_connect import birrt 
from manipulation.primitives.inverse_kinematics import *
from manipulation.motion.trajectories import *
from manipulation.constants import *
from manipulation.motion.trajectories import TrajTrajectory, PathTrajectory

import sys
import time
from time import sleep
import pickle
sys.path.append('../mover_library/')
from operator_utils.grasp_utils import solveTwoArmIKs
from operator_utils.grasp_utils import compute_two_arm_grasp,translate_point,\
                                       compute_Tee_at_given_Ttool
from samplers import *
from utils import pick_obj,place_obj,compute_occ_vec
from conveyor_belt_problem import two_tables_through_door
from data_load_utils import load_place_data,convert_collision_vec_to_one_hot
from conveyor_belt_problem import two_tables_through_door,sample_placement_using_gen
from utils import compute_occ_vec,set_robot_config,remove_drawn_configs,\
                  draw_configs,clean_pose_data,draw_robot_at_conf,\
                  pick_obj,place_obj

GRAB_SLEEP_TIME=0.05
PI = np.pi

def search_episode(problem,max_exp=np.inf,max_time=np.inf,Gplace=None,Gpick=None):
  is_adv_network = Gplace.__class__.__module__ != 'gps.ConstraintGPS'
  initial_state  = problem['initial_state']
  OBSTACLES      = problem['obstacles']
  OBJECTS        = problem['objects']
  loading_region = problem['loading_region']
  all_region     = problem['all_region']
  env            = problem['env']

  robot = env.GetRobots()[0]
  leftarm_manip = robot.GetManipulator('leftarm')
  rightarm_manip = robot.GetManipulator('rightarm')
  rightarm_torso_manip = robot.GetManipulator('rightarm_torso')

  initial_time = time.time()
  max_placements = 0
  init_base_conf = np.array([0,1.05,0])
  robot = env.GetRobots()[0]

  queue = PriorityQueue()
  PERSISTENT = True  # Consider the initial state again
  RECONSIDER = False  
  init_state_pval=len(OBJECTS)+1
  queue.push(init_state_pval, (initial_state,None,None)) # TODO - put the nodes back in

  # number of objects placed after using x amount of time
  score_time_list = []
  nodes = []
  goal_state, last_node = None, None
  max_placements= 0 
  pred_time=0

  n_place_samples = 0

  while (goal_state is None and not queue.empty())\
       and ((len(nodes) < max_exp) and (time.time() - initial_time) < max_time):
    # print times
    if max_time != np.inf:
      print time.time()-initial_time, max_time,' time/max_time'
    if max_exp != np.inf:
      print len(nodes), max_exp,time.time()-initial_time,' exped/max_exp'

    state,sample,parent = queue.pop() 
    saver,placements = state

    # add a node 
    node = TreeNode(state,\
                    sample=sample,\
                    parent=parent,\
                    rwd = len(placements))
    node.goal_node_flag = False
    node.pred_time = 0
    nodes += [node]

    # restore the environment
    sleep(GRAB_SLEEP_TIME)
    saver.Restore() 

    print max_placements, 'rwd' 

    # keep track of how many objects have been placed
    if max_placements < len(placements):
      max_placements = len(placements)
    time_used = time.time()-initial_time
    score_time_list.append([time_used,max_placements])

    # sample K actions
    n_tries = 5
    n_actions_per_state = 3
    n_actions = 0

    n_gen=100

    curr_obj = OBJECTS[len(placements)] # fixed object order
    # time to place if my arms are not folded
    place_precond = not np.all( np.isclose(leftarm_manip.GetArmDOFValues(),FOLDED_LEFT_ARM) )
    if place_precond is True:
      n_place_samples += 1
      for ntry in range(n_tries):
        sleep(GRAB_SLEEP_TIME)
        #print "restoring"
        saver.Restore()  #NOTE: when you restore, getgrabbed gets destroyed
        #print "restored"
        robot.Grab(curr_obj)
        #print "grabbed"

        place = {}
        if Gplace is None:
          place_obj_pose,place_robot_pose,path = sample_placement(env,curr_obj,robot,loading_region,all_region)
        else: 
          key_configs = Gplace.key_configs
          c_data = compute_occ_vec(key_configs,robot,env)[None,:]*1  
          scaled_c = convert_collision_vec_to_one_hot(c_data)
          c_data = np.tile(scaled_c,(n_gen,1,1))
        
          if is_adv_network:
            zvals = np.random.normal(size=(n_gen,Gplace.dim_z)).astype('float32')
            stime=time.time()
            Gpred = Gplace.a_gen.predict( [zvals,c_data] )
            p_samples = Gplace.x_scaler.inverse_transform(Gpred)
            pred_time+=time.time()-stime
          else:
            pass
            #Gpred = Gplace.policy.predict( c_data[:,:,:,None] )
            #noise = np.random.normal(size=(n_gen,Gplace.dim_action))  
            #p_samples = Gpred+noise
            # TODO: continue generating data from a policy
            #p_samples = Gplace.x_scaler.inverse_transform(Gpred)+p_samples
          #print 'prediction time',time.time()-stime

          VISUALIZE=True
          if VISUALIZE:
            draw_configs(p_samples,env,name='conf',transparency=0.5)
            sleep(GRAB_SLEEP_TIME)
            robot.Release(curr_obj)
            #with curr_obj:
            #  uniform_samples = np.array([randomly_place_in_region(env,curr_obj,loading_region)\
            #                       for _ in range(100)])
            #robot.Grab(curr_obj)
            #draw_configs(configs=uniform_samples,env=env,name='unif_conf',\
            #              colors=(0,0,1),transparency=0.5)
            if env.GetViewer() is None:     
              env.SetViewer('qtcoin')
            import pdb;pdb.set_trace()
            remove_drawn_configs('conf',env)
            remove_drawn_configs('unif_conf',env)
          
          
          place_obj_pose,place_robot_pose,path = sample_placement_using_gen(env,curr_obj,\
                                                                          robot,p_samples,\
                                                                          loading_region,\
                                                                          all_region)
        if place_obj_pose is None:
          continue
        else:
          place['place_base_pose'] = place_robot_pose
          place['place_obj_pose']  = place_obj_pose
          place['path']            = path
          place['obj']             = curr_obj.GetName()

          set_robot_config( place_robot_pose,robot)
          place_obj( curr_obj,robot,FOLDED_LEFT_ARM,leftarm_manip,rightarm_manip)
          set_robot_config(init_base_conf,robot) 
          sleep(GRAB_SLEEP_TIME)

          new_saver = DynamicEnvironmentStateSaver(env) # create a new saver for subsequent state
          new_state = (new_saver,placements+[place_obj_pose]) # collisions are preserved 
          new_placements = placements+[place_obj_pose]
          new_state_pval = len(OBJECTS) - len(new_placements) 
          if len(new_placements)==len(OBJECTS):
            print "Success"
            goal_node = TreeNode(new_state,sample=place,parent=node,rwd=len(OBJECTS))
            goal_node.goal_node_flag = True
            nodes += [goal_node]
            score_time_list.append([time.time()-initial_time,len(OBJECTS)])
            return nodes,score_time_list,n_place_samples
          #print 'pushing to queue'
          queue.push(new_state_pval, (new_state, place, node)) # push subsequent states
          #print 'pushed!'
          n_actions+=1
          if n_actions >= n_actions_per_state:
            break
    else:
      for ntry in range(n_tries):
        sleep(GRAB_SLEEP_TIME)
        saver.Restore()  #NOTE: when you restore, getgrabbed gets destroyed
        pick = {}
        pick_base_pose,grasp_params,g_config = sample_pick( curr_obj,robot,env,all_region )
        set_robot_config( pick_base_pose,robot)
        pick_obj( curr_obj,robot,g_config,leftarm_manip,rightarm_torso_manip )
        set_robot_config(init_base_conf,robot)

        pick['pick_base_pose'] = pick_base_pose
        pick['grasp_params']   = grasp_params
        pick['g_config']       = g_config
        pick['obj']            = curr_obj.GetName()
        
        sleep(GRAB_SLEEP_TIME)
        new_saver = DynamicEnvironmentStateSaver(env) 
        new_state = (new_saver,placements) # collisions are preserved 
        new_state_pval = len(OBJECTS)-len(placements)  # prefer the larger number of placements
        queue.push(new_state_pval, (new_state, pick, node)) # push subsequent states
        n_actions+=1
        if n_actions >= n_actions_per_state:
          break
    if queue.empty():
      queue.push(init_state_pval, (initial_state,None,None)) 
  return nodes,score_time_list,n_place_samples,rwd_time_list

def check_if_train_file_exists():
  n_episode= sys.argv[1]
  if os.path.isfile('./train_data/'+str(n_episode)+'.pkl'): 
    print 'already have'
    return True
  return False

def delete_kin_body_objects_for_pickling( nodes ):
  nodes_to_save = [n for n in nodes]
  for n in nodes_to_save: 
    n.state = n.state[1]
  return nodes_to_save

def create_problem_and_env():
  env=Environment()
  problem = two_tables_through_door(env)
  return problem,env

def save_train_data(problem_data,nodes_data,rwd_time_list):
  n_episode= sys.argv[1]
  nodes_to_save = delete_kin_body_objects_for_pickling( nodes_data )
  pickle.dump({ 'obst_shapes':problem_data['obst_shapes'],\
                'obst_poses':problem_data['obst_poses'],\
                'obj_shapes':problem_data['obj_shapes'],\
                'obj_poses':problem_data['obj_poses'],
                'nodes':nodes_to_save,\
                'rwd_time_list':rwd_time_list},\
                open('./train_data/'+str(n_episode)+'.pkl','wb'))

def collect_data_from_episode():
  if check_if_train_file_exists(): return
  problem,env = create_problem_and_env()
  nodes,rwd_time_list,_ = search_episode(problem,max_exp=50)
  RaveDestroy()
  env.Destroy()
  return problem,nodes,rwd_time_list

if __name__=='__main__':
  problem_data,nodes_data,rwd_time_list = collect_data_from_episode()
  save_train_data(problem_data,nodes_data,rwd_time_list)


