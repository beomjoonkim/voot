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

def activate_arms_torso_base(robot):
  leftarm_manip_indices        = robot.GetManipulator('leftarm').GetArmIndices().tolist()
  rightarm_torso_manip_indices = robot.GetManipulator('rightarm_torso').GetArmIndices().tolist()
  robot.SetActiveDOFs(leftarm_manip_indices+rightarm_torso_manip_indices, \
                      DOFAffine.X|DOFAffine.Y|DOFAffine.RotationAxis,[0,0,1])

def pick( obj_name,env,robot,all_region ):
  leftarm_manip = robot.GetManipulator('leftarm')
  rightarm_manip = robot.GetManipulator('rightarm')

  obj = env.GetKinBody(obj_name)
  pbase,grasp,g_config = sample_pick(obj,robot,env,all_region)
  print pbase is None
  pick_obj(obj,robot,g_config,leftarm_manip,rightarm_manip) 
 
  

def main():
  env=Environment()
  env.Load('env.xml')
  robot = env.GetRobots()[0]
  #set_default_robot_config(robot)
  robot_initial_pose = get_pose(robot)
  leftarm_manip = robot.GetManipulator('leftarm')
  robot.SetActiveManipulator('leftarm')

  ikmodel1 = databases.inversekinematics.InverseKinematicsModel(robot=robot, \
                                  iktype=IkParameterization.Type.Transform6D, \
                                  forceikfast=True, freeindices=None, \
                                  freejoints=None, manip=None)
  if not ikmodel1.load():
    ikmodel1.autogenerate()

  rightarm_manip = robot.GetManipulator('rightarm')
  rightarm_torso_manip = robot.GetManipulator('rightarm_torso')
  robot.SetActiveManipulator('rightarm_torso')
  manip = robot.GetActiveManipulator()
  ee    = manip.GetEndEffector()
  ikmodel2 = databases.inversekinematics.InverseKinematicsModel(robot=robot, \
                                iktype=IkParameterization.Type.Transform6D, \
                                forceikfast=True, freeindices=None, \
                                freejoints=None, manip=None)
  if not ikmodel2.load():
    ikmodel2.autogenerate()

  set_config(robot,FOLDED_LEFT_ARM,robot.GetManipulator('leftarm').GetArmIndices())
  set_config(robot,mirror_arm_config(FOLDED_LEFT_ARM),\
             robot.GetManipulator('rightarm').GetArmIndices())

  robot_initial_config = np.array([-1,1,0])
  set_robot_config(robot_initial_config,robot)
  env.SetViewer('qtcoin')

  target = env.GetKinBody('target')
  width = 0.1  #np.random.rand(1)*(max_width-min_width)+min_width
  length = 0.8 #np.random.rand(1)*(max_length-min_length) + min_length
  height = 1.0   #np.random.rand(1)*(max_height-min_height)+min_height
  new_body = box_body(env,width,length,height,\
                      name='obst',\
                      color=(0, 5, 1))
  trans = np.eye(4); 
  trans[2,-1] = 0.075
  #env.Add(new_body); 
  #new_body.SetTransform(trans)
 
  all_region = region = AARegion('all_region',((-2.51,2.51),(-2.51,2.51)),\
                                    z=0.0001,color=np.array((1,1,0,0.25)))

  activate_arms_torso_base(robot)
  # Try picking different objects
  pick_obj = lambda obj_name: pick(obj_name,env,robot,all_region)
  import pdb;pdb.set_trace()
  pbase,grasp,g_config = sample_pick(env.GetRobot('valkyrie'),robot,env,all_region)
  pick_obj(env.GetRobot('valkyrie'),robot,g_config,leftarm_manip,rightarm_manip) 

  import pdb;pdb.set_trace()
  path,tpath,status = get_motion_plan(robot,robot_xytheta,env,maxiter=10,n_node_lim=5000)
 
  import pdb;pdb.set_trace()
  sample_pick(target,robot,env,all_region)
  import pdb;pdb.set_trace()


if __name__=='__main__':
  main()


