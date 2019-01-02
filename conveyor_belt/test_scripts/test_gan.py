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
from manipulation.inverse_reachability.inverse_reachability import get_custom_ir, get_base_generator
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
sys.path.append('../../mover_library/')
sys.path.append('../')
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
from operator_utils.grasp_utils import solveTwoArmIKs
from operator_utils.grasp_utils import compute_two_arm_grasp,translate_point,\
                                       compute_Tee_at_given_Ttool
from samplers import *
from conveyor_belt_problem import two_tables_through_door

from data_load_utils import load_place_data
from PlaceGAN import PlaceGAN
import tensorflow as tf
GRAB_SLEEP_TIME=0.05

PI = np.pi

def compute_occ_vec(key_configs,robot,env):
  occ_vec = []
  with robot:
    for config in key_configs:
      set_robot_config(config,robot)
      occ_vec.append( env.CheckCollision(robot) )
  return np.array(occ_vec)


def create_custom_env():
  env=Environment()
  problem = two_tables_through_door(env)
  return problem

def define_generator(weight_f,n_data,n_trial,dim_data,dim_context,dim_konf):
  parent_dir = '../place_gan/n_data_'+str(n_data)
  trial_dir = parent_dir + '/n_trial_' + str(n_trial)
  scaler_dir = trial_dir + '/scalers/'
  train_results_dir = trial_dir + '/train_results/'
  test_results_dir = trial_dir +'/test_results/'
  if not os.path.exists(test_results_dir):
    os.mkdir(test_results_dir)

  sess = tf.Session()
  x_scaler = pickle.load( open(scaler_dir+'x_scaler.p','r') )
  c_scaler = pickle.load( open(scaler_dir+'c_scaler.p','r') )
  gan = PlaceGAN(sess,dim_data,dim_context,dim_konf,\
                   x_scaler=x_scaler,c_scaler=c_scaler,save_folder=train_results_dir)  
  gan.load_offline_weights(weight_f)
  return gan
def pick_obj(obj,robot,g_configs,left_manip,right_manip):
  set_config(robot, g_configs[0],left_manip.GetArmIndices())
  set_config(robot, g_configs[1],right_manip.GetArmIndices())
  sleep(GRAB_SLEEP_TIME)
  robot.Grab(obj)

def place_obj( obj,robot,arm_config,leftarm_manip,rightarm_manip):
  sleep(GRAB_SLEEP_TIME)
  robot.Release(obj)
  set_config(robot,FOLDED_LEFT_ARM,leftarm_manip.GetArmIndices())
  set_config(robot,mirror_arm_config(FOLDED_LEFT_ARM),\
              rightarm_manip.GetArmIndices())

def search_episode(problem,max_exp=np.inf,max_time=np.inf,Gplace=None,Gpick=None):
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
  rwd_time_list = []
  nodes = []
  goal_state, last_node = None, None
  max_placements= 0 
  pred_time=0

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
    rwd_time_list.append([time_used,max_placements])

    # sample K actions
    n_tries = 5
    n_actions_per_state = 3
    n_actions = 0

    n_gen=100
    konf =  np.tile(Gplace.scaled_k,(n_gen,1,1,1))

    curr_obj = OBJECTS[len(placements)] # fixed object order
    # time to place if my arms are not folded
    place_precond = not np.all( np.isclose(leftarm_manip.GetArmDOFValues(),FOLDED_LEFT_ARM) )
    if place_precond is True:
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
          occ_vec = compute_occ_vec(key_configs,robot,env)[None,:]*1
          scaled_c = Gplace.c_scaler.transform(occ_vec)
          scaled_c = scaled_c.reshape((1,len(key_configs),1))
          scaled_c = np.tile(scaled_c,(n_gen,1,1))
          zvals = np.random.normal(size=(n_gen,Gplace.dim_z)).astype('float32')
          stime=time.time()
          Gpred = Gplace.a_gen.predict( [zvals,konf,scaled_c] )
          pred_time+=time.time()-stime
          print 'prediction time',time.time()-stime
          p_samples = Gplace.x_scaler.inverse_transform(Gpred)
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
          #print "robot to place base pose"
          set_robot_config( place_robot_pose,robot)
          #print "placeobj"
          place_obj( curr_obj,robot,FOLDED_LEFT_ARM,leftarm_manip,rightarm_manip)
          #print "robot_toinit"
          set_robot_config(init_base_conf,robot) # NOTE: I am not planning back to the initial pose

          #print 'place declaring new environment. ntry= ' +str(ntry)
          sleep(GRAB_SLEEP_TIME)
          new_saver = DynamicEnvironmentStateSaver(env) # create a new saver for subsequent state
          #print 'done'
          new_state = (new_saver,placements+[place_obj_pose]) # collisions are preserved 
          new_placements = placements+[place_obj_pose]
          new_state_pval = len(OBJECTS) - len(new_placements) 
          if len(new_placements)==len(OBJECTS):
            print "Success"
            goal_node = TreeNode(new_state,sample=place,parent=node,rwd=len(OBJECTS))
            goal_node.goal_node_flag = True
            nodes += [goal_node]
            rwd_time_list.append([time.time()-initial_time,len(OBJECTS)])
            return nodes,rwd_time_list,pred_time
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
        
        #print 'pick new environment. ntry= ' +str(ntry)
        sleep(GRAB_SLEEP_TIME)
        new_saver = DynamicEnvironmentStateSaver(env) 
        #print 'done'
        new_state = (new_saver,placements) # collisions are preserved 
        new_state_pval = len(OBJECTS)-len(placements)  # prefer the larger number of placements
        queue.push(new_state_pval, (new_state, pick, node)) # push subsequent states
        n_actions+=1
        if n_actions >= n_actions_per_state:
          break
    if queue.empty():
      queue.push(init_state_pval, (initial_state,None,None)) 
  return nodes, rwd_time_list,pred_time


def main():
  n_data=5000
  n_trial=0

  # load key configs
  data = load_place_data( parent_dir='../place_gan/',\
                          proc_train_data_dir='../processed_train_data/',\
                          n_data=5000,n_trial=0)
  scaled_x = data['x']
  scaled_c = data['c']
  scaled_k = data['k']
  x_scaler = data['x_scaler']
  c_scaler = data['c_scaler']
  k_scaler = data['k_scaler']
  key_configs = data['key_configs']

  dim_data    = np.shape(scaled_x)[1]
  dim_context = (np.shape(scaled_c)[1],1) # single channel
  dim_konf = (np.shape(scaled_k)[1],np.shape(scaled_k)[2],1)
  weight_f = 'a_genepoch_40_3000_Dtrue_3.32202_Dfake_-0.676648.h5' # 3.58


  place_gan = define_generator(weight_f,n_data,n_trial,dim_data,dim_context,dim_konf)   
  n_gen=100
  place_gan.key_configs=key_configs


  if len(sys.argv) < 2:
    for pidx in range(100):
      print "New Problem"
      if os.path.isfile('./test_results/'+str(pidx)+'.pkl' ):
        print 'exists'
        continue
      problem = create_custom_env()
      #problem['env'].SetViewer('qtcoin')
      _,rwd_time,pred_time=search_episode(problem,max_exp=50,max_time=np.inf,Gplace=place_gan)
      #import pdb;pdb.set_trace()

      problem['env'].Destroy()  
      RaveDestroy()
      if rwd_time is not None:
        pickle.dump([rwd_time,pred_time],open('./test_results/'+str(pidx)+'.pkl','wb'))
  else:
    pidx = sys.argv[1]
    if os.path.isfile('./test_results/'+str(pidx)+'.pkl' ):
      print 'exists'
      return
    problem = create_custom_env()
    _,rwd_time,pred_time=search_episode(problem,max_exp=50,max_time=np.inf,Gplace=place_gan)
    problem['env'].Destroy()  
    RaveDestroy()
    if rwd_time is not None:
      pickle.dump([rwd_time,pred_time],open('./test_results/'+str(pidx)+'.pkl','wb'))
    

if __name__=='__main__':
  main()


