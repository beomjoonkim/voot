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
from samplers import *

def two_tables_through_door(env): # Previously 4, 8
  env.Load('env.xml')
  robot = env.GetRobots()[0]
  set_default_robot_config(robot)
  region = create_region(env, 'goal', ((-1, 1), (-.3, .3)), \
                         'floorwalls', color=np.array((0, 0, 1, .25)))

  set_config(robot,FOLDED_LEFT_ARM,robot.GetManipulator('leftarm').GetArmIndices())
  set_config(robot,mirror_arm_config(FOLDED_LEFT_ARM),\
             robot.GetManipulator('rightarm').GetArmIndices())
  

  # left arm IK
  robot.SetActiveManipulator('leftarm')
  manip = robot.GetActiveManipulator()
  ee    = manip.GetEndEffector()
  ikmodel1 = databases.inversekinematics.InverseKinematicsModel(robot=robot, \
                                iktype=IkParameterization.Type.Transform6D, \
                                forceikfast=True, freeindices=None, \
                                freejoints=None, manip=None)
  if not ikmodel1.load():
    ikmodel1.autogenerate()

  # right arm torso IK
  robot.SetActiveManipulator('rightarm_torso')
  manip = robot.GetActiveManipulator()
  ee    = manip.GetEndEffector()
  ikmodel2 = databases.inversekinematics.InverseKinematicsModel(robot=robot, \
                                iktype=IkParameterization.Type.Transform6D, \
                                forceikfast=True, freeindices=None, \
                                freejoints=None, manip=None)
  if not ikmodel2.load():
    ikmodel2.autogenerate()

  # obj definitions
  min_height = 0.4
  max_height = 1

  min_width = 0.2
  max_width = 0.6

  min_length = 0.2
  max_length = 0.6

  # loading areas
  #rightmost one
  init_loading_region = AARegion('init_loading_area',((-2.51,-0.81),(-2.51,-1)),z=0.0001,color=np.array((1,0,1,0.25)))
  init_loading_region.draw(env)
  init_loading_region2 = AARegion('init_loading_area2',((-2.51,-0.81),(1.7,2.6)),z=0.0001,color=np.array((1,0,1,0.25)))
  init_loading_region2.draw(env)
  init_loading_region3 = AARegion('init_loading_area3',((-1.3,-0.81),(-1,0)),z=0.0001,color=np.array((1,0,1,0.25)))
  init_loading_region3.draw(env)
  init_loading_region4 = AARegion('init_loading_area4',((-2.51,-2),(-1,0)),z=0.0001,color=np.array((1,0,1,0.25)))
  init_loading_region4.draw(env)
  loading_regions =[init_loading_region,init_loading_region2,\
                    init_loading_region3,init_loading_region4]

  loading_region = AARegion('loading_area',((-2.51,-0.81),(-2.51,2.51)),z=0.0001,color=np.array((1,1,0,0.25)))
  loading_region.draw(env)

  # converyor belt region
  conv_x = 2
  conv_y = 1
  conveyor_belt = AARegion('conveyor_belt',((-1+conv_x,10*max_width+conv_x),(-0.4+conv_y,0.5+conv_y)),z=0.0001,color=np.array((1,0,0,0.25)))
  conveyor_belt.draw(env)

  all_region = AARegion('all_region',((-2.51,10*max_width+conv_x),(-3.51,3.51)),z=0.0001,color=np.array((1,1,0,0.25)))

  """
  obj1 = box_body(env,0.5,0.5,0.5,\
                      name='obst1',\
                      color=(0, 1, 1))
  env.Add(obj1)
  obj2 = box_body(env,0.5,0.5,0.5,\
                      name='obst2',\
                      color=(0, 1, 1))
  env.Add(obj2)
  set_point(obj1,[-1,-1,0.75])

  set_point(obj1,[-1.9,-0.5,0.01])
  set_point(obj2,[-1.,-0.5,0.01])
  set_point(obj2,[-1,0.7,0.01])
  """


  NUM_OBSTACLES=4
  OBSTACLES = []
  obstacle_poses = {}
  obstacle_shapes = {}
  i = 0
  for i in range(NUM_OBSTACLES):
    width = np.random.rand(1)*(max_width-min_width)+min_width
    length = np.random.rand(1)*(max_length-min_length) + min_length
    height = np.random.rand(1)*(max_height-min_height)+min_height
    trans = np.eye(4); 
    trans[2,-1] = 0.075
    new_body = box_body(env,width,length,height,\
                        name='obj%s'%i,\
                        color=(0, (i+.5)/NUM_OBSTACLES, 0))
    env.Add(new_body); new_body.SetTransform(trans)
    xytheta =  randomly_place_in_region(env,new_body,loading_regions[np.random.randint(4)])
    
    if not(xytheta is None):
      obstacle_shapes['obst%s'%len(OBSTACLES)] = [width[0],length[0],height[0]] 
      obstacle_poses['obst%s'%len(OBSTACLES)] = xytheta
      OBSTACLES.append(new_body)
    else:
      env.Remove(new_body)
  goal_base_pose = np.array([-2,-2,5*PI/4])
  robot.SetActiveDOFs([],DOFAffine.X|DOFAffine.Y|DOFAffine.RotationAxis,[0,0,1])
  import pdb;pdb.set_trace()
  n_node_lim_list = [3000,4000,5000,6000,7000]#,8000,9000,1000]
  stime=time.time()
  n_node_lim=np.inf
  for n_node_lim in n_node_lim_list:
    path,tpath2,status2=get_motion_plan(robot,goal_base_pose,env,maxiter=20,n_node_lim=n_node_lim)
    if status2 is "HasSolution":
      print n_node_lim
      break
  print time.time()-stime
  import pdb;pdb.set_trace()
  set_robot_config(goal_base_pose,robot)

  """
  NUM_OBJECTS=5
  OBJECTS=[]
  obj_shapes = {}
  obj_poses = {}
  for i in range(NUM_OBJECTS):
    width = np.random.rand(1)*(max_width-min_width)+min_width
    length = np.random.rand(1)*(max_width-min_length) + min_length
    height = np.random.rand(1)*(max_height-min_height)+min_height
    new_body = box_body(env,width,length,height,\
                        name='obj%s'%i,\
                        color=(0, (i+.5)/NUM_OBJECTS, 0))
    trans = np.eye(4); 
    trans[2,-1] = 0.075
    env.Add(new_body); new_body.SetTransform(trans)
    xytheta = randomly_place_in_region(env,new_body,conveyor_belt,np.array([0]))
    OBJECTS.append(new_body)
    obj_shapes['obj%s'%i] =  [width[0],length[0],height[0]]
    obj_poses['obj%s'%i] =  xytheta
  """


env=Environment()
env.SetViewer('qtcoin')
two_tables_through_door(env)

