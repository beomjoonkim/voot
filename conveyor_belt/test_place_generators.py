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
sys.path.append('../mover_library/')
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
from generators.PlaceSOAP import PlaceSOAP
#from generators.PlaceGAN import PlaceGAN
import tensorflow as tf
from utils import determine_best_weight_path_for_given_n_data,get_best_weight_file
from train_scripts.train_algo import create_soap
#from train_gan import create_gan
from data_load_utils import load_place_data,convert_collision_vec_to_one_hot
from collect_data import search_episode,create_problem_and_env

GRAB_SLEEP_TIME=0.05
PI = np.pi
USESOAP=True

def create_generator(train_results_dir):
  #TODO Fix this awkard set up of train results dir
  n_data  = int(sys.argv[1])
  n_trial = sys.argv[2]
  if USESOAP:
    generator = create_soap(n_data,n_trial)  
    #weight_f = get_best_weight_file( train_results_dir )
    #weight_f='a_genepoch_980_3000_Dtrue_3.40407_Dfake_3.40407.h5'
    weight_f = 'a_genlambda_1epoch_40_2.0.h5'
    generator.a_gen.load_weights(train_results_dir+weight_f)
  else:
    generator = create_gan(n_data,n_trial)
    weight_f = 'a_gen_1_.h5'

  # for compilation and debugging purpose
  n_gen  = n_data
  zvals  = np.random.normal(size=(n_gen,generator.dim_z)).astype('float32')
  c_data = np.zeros( (n_gen,890,2))
  Gpred  = generator.a_gen.predict( [zvals,c_data] )
  Gpred  = generator.x_scaler.inverse_transform(Gpred)
  return generator

def make_test_results_dir():
  n_data  = int(sys.argv[1])
  n_trial = sys.argv[2]

  if USESOAP:
    #assert(n_trial=='binary_collision')
    parent_dir =  './place_soap/n_data_'+str(n_data)
  else:
    parent_dir = './place_gan/n_data_'+str(n_data)
    
  trial_dir = parent_dir + '/n_trial_' + str(n_trial)
  train_results_dir = trial_dir + '/train_results/'
  test_results_dir = trial_dir +'/test_results/'
  if not os.path.exists(test_results_dir):
    os.mkdir(test_results_dir)
  return train_results_dir, test_results_dir

def check_if_pidx_already_tested(test_results_dir):
  pidx = sys.argv[3]
  assert not os.path.isfile(test_results_dir+'/'+str(pidx)+'.pkl'),'already tested'

def test_specified_pidx():
  # TODO: better way: implement a class
  train_results_dir,test_results_dir = make_test_results_dir()
  check_if_pidx_already_tested(test_results_dir)
  generator = create_generator(train_results_dir)

  problem,env = create_problem_and_env()
  _,rwd_time,n_place_samples=search_episode(problem,max_exp=50,max_time=np.inf,Gplace=generator)
  env.Destroy()  
  RaveDestroy()
  return [rwd_time,n_place_samples]

def save_test_results( test_results ):
  pidx = sys.argv[3]
  if test_results[0] is not None:
    pickle.dump(test_results,open(test_results_dir+'/'+str(pidx)+'.pkl','wb'))

if __name__=='__main__':
  assert len(sys.argv) == 4, "we need: n_data,n_trial,pidx"
  test_results = test_specified_pidx()
  save_test_results(test_results)


