# search related libs
from TreeNode import *
from generators.PlaceUniform import PlaceUnif
from misc.priority_queue import Stack, Queue, FILOPriorityQueue, PriorityQueue

import sys
sys.path.append('../mover_library/')
from samplers import *
from utils import *

import time

def add_node(nodes,state,sample,parent,rwd):
  node = TreeNode(state,\
                  sample=sample,\
                  parent=parent,\
                  rwd = rwd)
  node.goal_node_flag = False
  node.pred_time = 0
  nodes += [node]
  return node

def create_new_state(env,placements):
  new_saver = DynamicEnvironmentStateSaver(env) 
  new_state = (new_saver,placements)             # collisions are preserved 
  return new_state

class FocusedSearch():
  def __init__(conv_belt,max_exp,policy):
    self.conv_belt = conv_belt
    self.max_exp = max_exp
    self.policy = policy
  
  def instantiate_samplers(self):
    env            = problem['env']
    robot = self.env.GetRobots()[0]

    for b in self.objects:
      pick_base_pose,g_params,_ = sample_pick(b,robot,env,all_region)
      picks[b.GetName()].append([pick_base_pose,g_params])

      obj_placement = self.policy.predict(env,\
                                       robot,\
                                       b,\
                                       loading_region,\
                                       all_region)[0]
      places[b.GetName()].append(obj_placement) 

  def setup_problem_params(self):
    problem        = self.conv_belt.problem
    self.initial_state  = problem['initial_state']
    self.obstacles      = problem['obstacles']
    self.objects   = problem['objects']
    self.loading_region = problem['loading_region']
    self.all_region     = problem['all_region']
    self.env            = problem['env']
    self.robot = env.GetRobots()[0]

  def search(self):
    # focused algorithm as implemented in stripstream paper 
    self.setup_problem_params()

    leftarm_manip = self.robot.GetManipulator('leftarm')
    rightarm_manip = self.robot.GetManipulator('rightarm')
    rightarm_torso_manip = self.robot.GetManipulator('rightarm_torso')

    initial_time = time.time()
    max_placements = 0
    init_base_conf = np.array([0,1.05,0])


    rwd_n_expanded_list = [] # number of objects placed after using x amount of time
    nodes = []
    max_placements= 0 

    # samplers - focused sampling allows us to define one for each object
    self.pick_sampler={};self.place_sampler={}
    for b in OBJECTS:
      self.pick_sampler[b.GetName()]  = []
      self.place_sampler[b.GetName()] = [] 

    while len(nodes)<max_exp:
      # instantiate samplers
      self.instantiate_samplers()
      # sample values
      sample() #



        


    # sample place operator instances for objects 1 to 5

    # check if they work 
    #   - discrete search, which amounts to checking if the paths for each objects
    #     are collision free

    # if one fails, re-sample. Do I try to use the past samples too?


  """
  while (goal_state is None and not queue.empty())\
       and (len(nodes) < max_exp):
    # print times
    if max_exp != np.inf:
      print len(nodes), max_exp,time.time()-initial_time,' exped/max_exp,time_used'

    state,sample,parent = queue.pop() 
    saver,placements = state

    curr_node = add_node(nodes,state,sample,parent,len(placements))

    # restore the environment
    saver.Restore() 

    print max_placements, 'rwd' 

    if max_placements < len(placements):
      max_placements = len(placements)

    rwd_n_expanded_list.append([len(nodes),max_placements])

    # sample K actions 
    n_tries = 5
    n_actions_per_state = 3
    n_actions = 0
    conv_belt.curr_obj = OBJECTS[len(placements)] # fixed object order

    # time to place if my arms are not folded
    place_precond = not np.all( np.isclose(leftarm_manip.GetArmDOFValues(),FOLDED_LEFT_ARM) )
    if place_precond:
      if conv_belt.v:
        saver.Restore()  
        grab_obj(robot,conv_belt.curr_obj)
        conv_belt.visualize_placements(policy)
      for ntry in range(n_tries):
        saver.Restore()  
        grab_obj(robot,conv_belt.curr_obj)

        place = {}
        place_robot_pose = sample_action( conv_belt,policy )
        has_path= conv_belt.check_action_feasible( place_robot_pose )
        if has_path:
          place['place_base_pose'] = place_robot_pose
          place['obj']             = conv_belt.curr_obj.GetName()

          set_robot_config( place_robot_pose,robot)
          place_obj( conv_belt.curr_obj,robot,FOLDED_LEFT_ARM,leftarm_manip,rightarm_manip)
          set_robot_config(init_base_conf,robot) 

          new_placements = placements+[place_robot_pose]
          new_state = create_new_state( env,new_placements )

          is_goal = len(new_placements)==len(OBJECTS)
          if is_goal:
            print "Success"
            #goal_node = TreeNode(new_state,sample=place,parent=node,rwd=len(OBJECTS))
            #goal_node.goal_node_flag = True
            #nodes += [goal_node]
            add_node(nodes,new_state,place,curr_node,len(OBJECTS))
            rwd_n_expanded_list.append([len(nodes),len(OBJECTS)])
            return nodes,rwd_n_expanded_list
          
          new_state_val = -compute_V( new_placements,conv_belt,policy ) # smaller the better
          print 'New state value is ',new_state_val
          import pdb;pdb.set_trace()
          queue.push(new_state_val, (new_state, place, curr_node)) # push subsequent states
          n_actions+=1
          if n_actions >= n_actions_per_state:
            break
    else:
      for ntry in range(n_tries):
        saver.Restore()  
        pick = conv_belt.apply_pick_action()
      
        new_state = create_new_state( env,placements )
        new_state_val = -compute_V( placements,conv_belt,policy ) # smaller the better
        queue.push(new_state_val, (new_state, pick, curr_node)) # push subsequent states
        n_actions+=1
        if n_actions >= n_actions_per_state:
          break

    if queue.empty():
      # persistency
      queue.push(init_state_pval, (initial_state,None,None)) 

  # What's the intuition in backpropagatig the values in AlphaZero?
  #  - At the end of the day, you want to return the action with maximum value
  #  - This value gets accurate as you do more roll-outs, and as you get near the end
  #  - of the game, because in the game of Go, your reward is 0 or 1 at the end
  # I probably don't need such update. All I am trying to do is to 
  # find a path to the goal. For the game of Go, we cannot know if we are at the
  # winning state - or a terminal state even, I think.
  # But can this help in getting a better estimate of the heuristic?
  # They do not have any heuristic at all, but we do.

  # The question is if the value function that we learn is better than 
  # the heuristic function.
  
  # heuristic function = T - n_objs_packed
  # Q fcn approximates sumR from the current state

  # which one is more accurate in terms of 
  # getting the state that is closer to the goal?
  
  # The heuristic function can be wrong if we are at the deadend
  # On the other hand, if we are at the deadend, Q function can detect that
  
  # Heuristic function is local; Q fcn is global

  # What if I do 1 and 0 for scoring trajectories?
  # Then it will be sample inefficient because
  # it will ignore the ones with that packed 4 objects
  # But this depends on how I defined "success"
  # If my goal is to maximize the number of objects packed, the current
  # reward function is correct. But if my goal is to pack more than 4 objects,
  # then I can do 0/1 reward on trajectories
  """
  return nodes, rwd_n_expanded_list



