from __future__ import print_function

import sys
sys.path.append('/home/beomjoon/Documents/Github/openrave_wrapper/')
sys.path.append('/home/beomjoon/Documents/Github/openrave_wrapper/manipulation/')
sys.path.append('/home/beomjoon/Documents/Github/AdversarialOptimization/conveyor_belot/')
#from manipulation.problems.fixed import ENVIRONMENTS_DIR
from manipulation.bodies.bodies import box_body 
from manipulation.problems.problem import *
from manipulation.bodies.bodies import get_name
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
from operator_utils.grasp_utils import compute_two_arm_grasp,\
                                       translate_point,\
                                       compute_Tee_at_given_Ttool
from samplers import *

# obj definitions
min_height = 0.4
max_height = 1

min_width = 0.2
max_width = 0.6

min_length = 0.2
max_length = 0.6

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
    """
    if max_time != np.inf:
      print time.time()-initial_time, max_time,' time/max_time'
    if max_exp != np.inf:
      print len(nodes), max_exp,time.time()-initial_time,' exped/max_exp'
    """
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

    print( max_placements, 'rwd' )

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
          place_obj_pose,place_robot_pose,path = sample_placement(env,curr_obj,robot,\
                                                                  loading_region,all_region)
        else: 
          key_configs = Gplace.key_configs
          c_data = compute_occ_vec(key_configs,robot,env)[None,:]*1  
          scaled_c = convert_collision_vec_to_one_hot(c_data)
          c_data = np.tile(scaled_c,(n_gen,1,1))
        
          if is_adv_network:    
            Gplace.generate(c_data,n_gen)
            """
            zvals = np.random.normal(size=(n_gen,Gplace.dim_z)).astype('float32')
            stime=time.time()
            Gpred = Gplace.a_gen.predict( [zvals,c_data] )
            p_samples = Gplace.x_scaler.inverse_transform(Gpred)
            pred_time+=time.time()-stime
            """
          else:
            p_samples = Gplace.predict( c_data[:,:,:,None],n_gen=n_gen )
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
            print( "Success" )
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

def sample_placement_using_gen( env,obj,robot,p_samples,
                                obj_region,robot_region,
                                GRAB_SLEEP_TIME=0.05):
  # This script, with a given grasp of an object,
  # - finding colfree obj placement within 100 tries. no robot at this point
  # - checking colfree ik solution for robot; if not, trial is over
  # - if above two passes, colfree path finding; if not, trial is over
  status = "Failed"
  path= None
  original_trans = robot.GetTransform()
  obj_orig_trans = obj.GetTransform()
  tried = []
  n_trials = 1 # try 5 different samples of placements with a given grasp
  T_r_wrt_o = np.dot( np.linalg.inv( obj.GetTransform()), robot.GetTransform())
  for _ in range(n_trials):
    #print 'releasing obj'
    sleep(GRAB_SLEEP_TIME)
    robot.Release(obj)
    robot.SetTransform(original_trans)
    obj.SetTransform(obj_orig_trans)
    #print 'released'
    # get robot pose wrt obj

    # sample obj pose
    #print 'randmly placing obj'
    #obj_xytheta = randomly_place_in_region(env,obj,obj_region) # randomly place obj
    inCollision = True
    np.random.shuffle(p_samples)

    for idx,obj_xytheta in enumerate(p_samples):
      if idx>100: break
      x = obj_xytheta[0]
      y = obj_xytheta[1]
      z = get_point(obj)[-1]
      set_point(obj,[x,y,z])
      th=obj_xytheta[2]
      set_quat( obj, quat_from_z_rot(th) )

      new_T_robot = np.dot( obj.GetTransform(),T_r_wrt_o) 
      robot.SetTransform(new_T_robot)

      inCollision = (check_collision_except(obj,robot,env))\
                    or (check_collision_except(robot,obj,env))
      inRegion = (robot_region.contains(robot.ComputeAABB())) and\
                 (obj_region.contains(obj.ComputeAABB()))
      if (not inCollision) and inRegion:
        break
    if inCollision or not(inRegion):
      break # if you tried all p samples and ran out, get new pick
    
    # compute the resulting robot transform
    sleep(GRAB_SLEEP_TIME)
    robot.Grab(obj)
    robot.SetTransform(new_T_robot)
    robot.SetActiveDOFs([],DOFAffine.X|DOFAffine.Y|DOFAffine.RotationAxis,[0,0,1])
    robot_xytheta=robot.GetActiveDOFValues()
    robot.SetTransform(original_trans)
    stime=time.time()
    for node_lim in [1000,5000,np.inf]:
      path,tpath,status = get_motion_plan(robot,\
                  robot_xytheta,env,maxiter=10,n_node_lim=node_lim)
      if path=='collision':
      #  import pdb;pdb.set_trace()
        pass
      if status == "HasSolution":  
        robot.SetTransform(new_T_robot)
        return obj_xytheta,robot_xytheta,path
      else:
        print('motion planning failed',tpath)
  sleep(GRAB_SLEEP_TIME)
  robot.Grab(obj)
  robot.SetTransform(original_trans)
  print("Returnining no solution" )
  return None,None,None

def create_obstacles(env,loading_regions):
  NUM_OBSTACLES=4
  OBSTACLES = []
  obstacle_poses = {}
  obstacle_shapes = {}
  i = 0
  while len(OBSTACLES) < NUM_OBSTACLES:
    width = np.random.rand(1)*(max_width-min_width)+min_width
    length = np.random.rand(1)*(max_length-min_length) + min_length
    height = np.random.rand(1)*(max_height-min_height)+min_height
    new_body = box_body(env,width,length,height,\
                        name='obst%s'%len(OBSTACLES),\
                        color=(0, (i+.5)/NUM_OBSTACLES, 1))
    trans = np.eye(4); 
    trans[2,-1] = 0.075
    env.Add(new_body); new_body.SetTransform(trans)
    xytheta =  randomly_place_in_region(env,new_body,\
               loading_regions[np.random.randint(len(loading_regions))])
    
    if not(xytheta is None):
      obstacle_shapes['obst%s'%len(OBSTACLES)] = [width[0],length[0],height[0]] 
      obstacle_poses['obst%s'%len(OBSTACLES)] = xytheta
      OBSTACLES.append(new_body)
    else:
      env.Remove(new_body)
  return OBSTACLES,obstacle_shapes,obstacle_poses

def create_objects(env,conveyor_belt):
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
  return OBJECTS,obj_shapes,obj_poses

def load_objects(env, obj_shapes, obj_poses, color):
  OBJECTS=[]
  i = 0
  nobj = len(obj_shapes.keys())
  for obj_name in obj_shapes.keys():
    xytheta             = obj_poses[obj_name]  
    width,length,height = obj_shapes[obj_name]
    quat = quat_from_z_rot(xytheta[-1])
    
    new_body = box_body(env,width,length,height,\
                        name=obj_name,\
                        color=np.array(color)/float(nobj-i))
    i+=1
    env.Add(new_body); 
    set_point(new_body,[xytheta[0],xytheta[1],0.075])
    set_quat(new_body,quat)
    OBJECTS.append(new_body)
  return OBJECTS
    
def two_tables_through_door(env,obj_shapes=None,obj_poses=None,
                            obst_shapes=None,obst_poses=None): 
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

  # loading areas
  init_loading_region = AARegion('init_loading_area',\
                        ((-2.51,-0.81),(-2.51,0)),\
                        z=0.0001,color=np.array((1,0,1,0.25)))
  init_loading_region.draw(env)
  init_loading_region2 = AARegion('init_loading_area2',\
                         ((-2.51,-0.81),(1.7,2.6)),\
                         z=0.0001,color=np.array((1,0,1,0.25)))
  init_loading_region2.draw(env)
  init_loading_region4 = AARegion('init_loading_area4',\
                         ((-2.51,-1.5),(-0.1,2)),\
                          z=0.0001,color=np.array((1,0,1,0.25)))
  init_loading_region4.draw(env)
  loading_regions =[init_loading_region,init_loading_region2,\
                    init_loading_region4]

  loading_region = AARegion('loading_area',\
                  ((-2.51,-0.81),(-2.51,2.51)),\
                  z=0.0001,color=np.array((1,1,0,0.25)))
  loading_region.draw(env)

  # converyor belt region
  conv_x = 2
  conv_y = 1
  conveyor_belt = AARegion('conveyor_belt',\
                  ((-1+conv_x,10*max_width+conv_x),\
                  (-0.4+conv_y,0.5+conv_y)),\
                  z=0.0001,color=np.array((1,0,0,0.25)))
  conveyor_belt.draw(env)

  all_region = AARegion('all_region',\
               ((-2.51,10*max_width+conv_x),(-3.51,3.51)),\
               z=0.0001,color=np.array((1,1,0,0.25)))


  if obj_shapes == None:
    OBJECTS,obj_shapes,obj_poses             = create_objects(env,conveyor_belt)
  else:
    OBJECTS   = load_objects(env,obj_shapes,obj_poses,color=(0,1,0))

  if obst_shapes == None:
    OBSTACLES,obst_shapes,obst_poses = create_obstacles(env,loading_regions)
  else:
    OBSTACLES = load_objects(env,obst_shapes,obst_poses,color=(0,0,1))

  initial_saver = DynamicEnvironmentStateSaver(env)
  initial_state = (initial_saver,[])
  init_base_conf = np.array([0,1.05,0])

  problem = {'initial_state':initial_state,\
             'obstacles':OBSTACLES,\
             'objects':OBJECTS,\
             'loading_region':loading_region,\
             'env':env,\
             'obst_shapes':obst_shapes,\
             'obst_poses':obst_poses,\
             'obj_shapes':obj_shapes,\
             'obj_poses':obj_poses,\
             'all_region':all_region,\
             'init_base_conf':init_base_conf}
  return problem # the second is for indicating 0 placed objs


