import numpy as np
from time import sleep

from data_load_utils import convert_collision_vec_to_one_hot

from conveyor_belt_problem import two_tables_through_door
from openravepy import *
from utils import compute_occ_vec,set_robot_config,remove_drawn_configs,\
                  draw_configs,clean_pose_data,draw_robot_at_conf,\
                  pick_obj,place_obj,check_collision_except

from manipulation.primitives.savers import DynamicEnvironmentStateSaver

import sys
sys.path.append('../mover_library/')
from samplers import sample_pick,get_motion_plan
import collect_data
import preprocess_place_data

## imports from CGR's lib
from manipulation.constants import FOLDED_LEFT_ARM


GRAB_SLEEP_TIME=0.1

class ConveyorBelt:
  def __init__(self):
    self.create_problem_and_env()

  def create_problem_and_env(self):
    self.env            = Environment()
    self.problem        = two_tables_through_door(self.env)
    self.objects        = self.problem['objects']
    self.robot          = self.env.GetRobots()[0]
    self.init_base_conf = np.array([0,1.05,0])
    self.all_region     = self.problem['all_region']
    self.loading_region = self. problem['loading_region']
    self.init_saver     = DynamicEnvironmentStateSaver(self.env)

  def reset_to_init_state( self ):
    self.robot.ReleaseAllGrabbed()
    self.init_saver.Restore()
    self.placements=[]

  def get_state(self,key_configs):
    # our state is represented with a key configuration collision vector
    c_data = compute_occ_vec(key_configs,self.robot,self.env)[None,:]*1  
    scaled_c = convert_collision_vec_to_one_hot(c_data)
    c_data = np.tile(scaled_c,(1,1,1))
    c_data = c_data[:,:,:,None]
    return c_data

  def apply_pick_action(self):
    robot    = self.robot
    env      = self.env
    
    leftarm_manip        = robot.GetManipulator('leftarm')
    rightarm_torso_manip = robot.GetManipulator('rightarm_torso')
    
    while True:
      pick_base_pose,grasp_params,g_config = sample_pick(self.curr_obj,robot,env,self.all_region)
      if pick_base_pose is None:
        continue
      
      set_robot_config( pick_base_pose,robot)
      pick_obj(self.curr_obj,robot,g_config,leftarm_manip,rightarm_torso_manip )
      set_robot_config(self.init_base_conf,robot)

      pick = {}
      pick['pick_base_pose'] = pick_base_pose
      pick['grasp_params']   = grasp_params
      pick['g_config']       = g_config
      pick['obj']            = self.curr_obj.GetName()
      
      return pick

  def check_action_feasible(self,action):
    robot    = self.robot
    env      = self.env
    #curr_obj = self.objects[len(self.placements)] # fixed object order
 
    place_obj_pose   = action[0,0:3]
    place_robot_pose = action[0,3:]

    with robot:
      status = ''
      set_robot_config( place_robot_pose,robot)
      inCollision = (check_collision_except(self.curr_obj,robot,env))\
                      or (check_collision_except(robot,self.curr_obj,env))
      inRegion = (self.all_region.contains(robot.ComputeAABB())) and\
                   (self.loading_region.contains(self.curr_obj.ComputeAABB()))

      #if inCollision: 
      #if not inRegion: print 'Out of collision'
      if (not inCollision) and inRegion:
        for node_lim in [1000,5000,np.inf]:
          path,tpath,status = get_motion_plan(robot,\
                    place_robot_pose,env,maxiter=10,n_node_lim=node_lim)
          if status=='HasSolution': break
    if status == "HasSolution":  
      return True
    else:
      return False
      
  def apply_place_action(self,action):
    robot    = self.robot
    env      = self.env
    leftarm_manip  = robot.GetManipulator('leftarm')
    rightarm_manip = robot.GetManipulator('rightarm')
    
    if self.check_action_feasible(action):
      place_robot_pose = action[0,3:]
      set_robot_config(place_robot_pose,robot)
      place_obj( self.curr_obj,robot,FOLDED_LEFT_ARM,leftarm_manip,rightarm_manip)
      return True
    return False
  
  def visualize_placements(self,policy,state):
    p_samples = policy.predict(state,n_samples=100)
    robot_xy  = p_samples[:,3:]
    draw_configs(robot_xy,self.env,name='conf',transparency=0.5)
    #sleep(GRAB_SLEEP_TIME)
    #self.robot.Release(self.curr_obj)
    #sleep(GRAB_SLEEP_TIME)
    #with curr_obj:
    #  uniform_samples = np.array([randomly_place_in_region(env,curr_obj,loading_region)\
    #                       for _ in range(100)])
    #robot.Grab(curr_obj)
    #draw_configs(configs=uniform_samples,env=env,name='unif_conf',\
    #              colors=(0,0,1),transparency=0.5)
    if self.env.GetViewer() is None:     
      self.env.SetViewer('qtcoin')
    raw_input('press a key to continue')
    remove_drawn_configs('conf',self.env)
    remove_drawn_configs('unif_conf',self.env)

  def execute_policy(self,policy,time_step_limit,visualize=False):
    self.placements      = []
    robot                = self.robot
    leftarm_manip        = robot.GetManipulator('leftarm')
    rightarm_torso_manip = robot.GetManipulator('rightarm_torso')
    traj                 = []
    states = []
    actions = []
    rewards = []


    while len(actions) < time_step_limit and len(self.placements) < len(self.objects):
      #print '%d/%d',len(states),time_step_limit
      place_precond = not np.all( np.isclose(leftarm_manip.GetArmDOFValues(),FOLDED_LEFT_ARM) )
      self.curr_obj = self.objects[len(self.placements)] # fixed object order

      if place_precond:
        states.append( self.get_state(policy.key_configs) ) # konf while holding the object
        self.state = states[-1]    # current state

        action = policy.predict(self.state) 
        actions.append(action);  # action performed in current state

        if visualize: self.visualize_placements(policy,self.state)
    
        is_action_success = self.apply_place_action(action) 
        if is_action_success:
          self.placements.append(action)
          rewards.append(1)
        else:
          rewards.append(0)
      else:
        self.apply_pick_action()
    traj = {'s':states,'a':actions,'r':rewards}
    return traj
  
  def execute_policy_with_planner(self,policy):
    raise NotImplemented
    nodes,_,_,_ = collect_data.search_episode( self.problem,max_exp=50,Gplace=self.policy )
    actions,_,states,_ = preprocess_place_data.process_search_episode()
    
    

  def apply_operator(self,op):
    pass


