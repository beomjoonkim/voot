from manipulation.problems.fixed import ENVIRONMENTS_DIR
from manipulation.bodies.bodies import box_body, place_xyz_body
from manipulation.problems.problem import *
from manipulation.bodies.bodies import get_name
from misc.functions import randomize
from misc.generators import take
from misc.numerical import INF
from manipulation.bodies.robot import set_default_robot_config
from manipulation.primitives.transforms import get_point, set_point, pose_from_quat_point, unit_quat
from misc.colors import get_color
from manipulation.constants import BODY_PLACEMENT_Z_OFFSET
from manipulation.primitives.utils import Pose

##TODO: Clean this
from manipulation.constants import PARALLEL_LEFT_ARM, REST_LEFT_ARM, HOLDING_LEFT_ARM, FOLDED_LEFT_ARM, FAR_HOLDING_LEFT_ARM, LOWER_TOP_HOLDING_LEFT_ARM,REGION_Z_OFFSET
from manipulation.regions import create_region, AARegion
from manipulation.bodies.bodies import randomly_place_region, place_body, place_body_on_floor
from manipulation.inverse_reachability.inverse_reachability import ir_base_trans
from manipulation.primitives.utils import mirror_arm_config
from manipulation.primitives.transforms import trans_from_base_values, set_pose, set_quat, \
  point_from_pose, axis_angle_from_rot, rot_from_quat, quat_from_pose, quat_from_z_rot,\
  get_pose,base_values_from_pose,pose_from_base_values, set_xy
from itertools import product
import numpy as np
import copy
import math

from manipulation.bodies.bounding_volumes import aabb_extrema, aabb_from_body, aabb_union
from manipulation.inverse_reachability.inverse_reachability import get_custom_ir, get_base_generator
from manipulation.bodies.robot import manip_from_pose_grasp
from manipulation.bodies.robot import get_active_arm_indices
from manipulation.grasps.grasps import FILENAME as GRASP_FILENAME, load_grasp_database
from manipulation.grasps.grasp_options import positive_hash, get_grasp_options
from manipulation.constants import GRASP_APPROACHES, GRASP_TYPES

from manipulation.bodies.bodies import geometry_hash
from manipulation.bodies.bounding_volumes import aabb_from_body
from manipulation.grasps.grasps import save_grasp_database, Grasp
from openravepy import *

import sys
sys.path.append('../mover_library/')
from samplers import *
from operator_utils.grasp_utils import solveTwoArmIKs,compute_two_arm_grasp
from misc.priority_queue import Stack, Queue,PriorityQueue
from TreeNode import *

import tensorflow as tf
from data_load_utils import load_place_data
from generators.PlaceSOAP import PlaceSOAP
from NAMO_problem import * #NAMO_problem,sample_pick,sample_placement,
from utils import get_robot_xytheta,get_body_xytheta,compute_occ_vec,clean_pose_data,\
                  draw_configs, check_collision_except, remove_drawn_configs
from utils import determine_best_weight_path_for_given_n_data,get_best_weight_file
from data_load_utils import convert_collision_vec_to_one_hot
from NAMO_env import NAMO
from forward_search import forward_dfs_search
from generators.Uniform import UniformPlace,UniformPick

SLEEPTIME=0.05

def get_sample_traj(traj_to_goal):
  sample_traj = []
  for step in traj_to_goal:
    sample_traj.append( step.sample )
  return sample_traj

def get_node_idx(ntarget,nodes):
  return np.where([ ntarget==n for n in nodes])[0][0]

def get_traj_idxs(node_idx,nodes,traj,traj_list):
  # inputs: root node idx and nodes, empty traj and traj list
  # outputs:
  #   traj: a list of a seuqence from a root node to a leaf node
  #   traj_list: a list of all sequences of idxs of to a leaf node
  node = nodes[node_idx]

  # a trajectory should end at a leaf node
  traj.append(node_idx)
  
  # get to the deepest level of the node
  for n in node.children:
    new_traj = copy.deepcopy(traj)
    child_idx = get_node_idx(n,nodes)
    child_traj,_ = get_traj_idxs(child_idx,nodes,new_traj,traj_list)

  if len(node.children) == 0:
    traj_list.append(traj)
  return traj,traj_list

def get_rl_data(traj):
  # Target object pose
  # Current object pose
  # Current robot pose
  for node in traj:
    saver = node[0][0]
    import pdb;pdb.set_trace()
    saver.Restore()
  
def create_RL_data(nodes):
  import pdb;pdb.set_trace()
  _,traj_idx_list = get_traj_idxs(0,nodes,[],[])
  trajs = [nodes[0] for for traj_idxs in traj_idx_list]
  import pdb;pdb.set_trace()
  
  for traj in traj_list:
    rl_traj = get_rl_data(traj)

def main(): 
  #test_results_dir = './train_data/'
  #if not os.path.exists(test_results_dir):
  #  os.mkdir(test_results_dir)

  problem = NAMO()
  problem.env.SetViewer('qtcoin')
  key_configs = pickle.load(open('./key_configs/key_configs.p','r'))
  place_pi = UniformPlace( problem.problem['env'], \
                           problem.problem['obj_region'], \
                           problem.problem['all_region'] )
  pick_pi  = UniformPick( problem.problem['env'], \
                          problem.problem['obj_region'], \
                          problem.problem['all_region'] )
  nodes,rwd_time = forward_dfs_search(problem,pick_pi,place_pi,max_exp=10)
  create_RL_data(nodes)
  import pdb;pdb.set_trace()
  

  """

  MAX_TIME = np.inf
  MAX_EXP  = 50

  if len(sys.argv) <= 1:
    for pidx in range(100):
      if os.path.isfile(test_results_dir+'/'+str(pidx)+'.pkl' ):
        print 'exists'
        continue

      env=Environment()
      problem = NAMO_problem()
      problem.env.SetViewer('qtcoin')
      import pdb;pdb.set_trace()

      rwd_time,n_place_samples,n_expanded = search_episode(problem,\
                                                           max_exp=MAX_EXP,\
                                                           max_time=MAX_TIME)
      problem['env'].Destroy()  
      RaveDestroy()
      if rwd_time is not None:
        pickle.dump([rwd_time,n_place_samples,n_expanded],open(test_results_dir+'/'+str(pidx)+'.pkl','wb'))
  else:
    idx = sys.argv[1]
    np.random.seed(int(idx))
    if os.path.isfile('./train_data/episode_'+str(idx)+'.pkl'):
      print 'exists'
      return
    env=Environment()
    #env.SetViewer('qtcoin')
    problem = NAMO_problem(env)
    obj_shapes = problem['obj_shapes']
    obj_poses  = problem['obj_poses']
    final_robot_base_pose = problem['base_pose']
    final_grasp = problem['grasp_params']
    original_path = problem['original_path']
    collided_objs = problem['collided_objs']
    target_obj = problem['target_obj']
    robot =  env.GetRobots()[0]

    DEBUG=False
    if DEBUG:
      draw_robot_at_path(original_path[0:-1:int(0.1*len(original_path))],robot,env)
      import pdb;pdb.set_trace()

    nodes,rwd_time_list = search_episode(problem,max_exp=MAX_EXP,max_time=MAX_TIME)
    problem['env'].Destroy()  
    nodes_to_save = [n for n in nodes]
    for n in nodes_to_save: 
      n.state = n.state[1]
    #import pdb;pdb.set_trace()
    pickle.dump({'obj_shapes':obj_shapes,\
                 'obj_poses':obj_poses,\
                 'final_robot_base_pose':final_robot_base_pose,
                 'final_grasp': final_grasp,\
                 'original_path':original_path,\
                 'collided_objs':collided_objs,\
                 'target_obj_name':target_obj.GetName(),\
                 'rwd_time_list':rwd_time_list,\
                 'nodes':nodes_to_save},\
                  open('./train_data/episode_'+str(idx)+'.pkl','wb'))
    RaveDestroy()
  """
 

if __name__ == '__main__':
  main()
    




  







