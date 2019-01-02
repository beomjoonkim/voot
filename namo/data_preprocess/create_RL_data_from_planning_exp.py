import os
import pickle
import copy
import numpy as np
import pickle
import time
import sys

from openravepy import *
from NAMO_env import NAMO
from preprocessing_utils import compute_fetch_vec
sys.path.append('../mover_library/')
from utils import compute_occ_vec,clean_pose_data,get_body_xytheta,\
                  convert_rel_to_abs_base_pose,visualize_path
from samplers import set_robot_config,compute_angle_to_be_set

def get_node_idx(ntarget,nodes):
  return np.where([ ntarget==n for n in nodes])[0][0]

def get_training_data_from_traj(traj,nodes,key_configs,NAMOenv):
  print "New trajectory"
  NAMOenv.reset_to_init_state()
  problem         = NAMOenv.problem
  obj_poses       = problem['obj_poses']
  target_obj_pose = obj_poses[problem['target_obj'].GetName()]
  init_state      = problem['initial_state'][0]
  
  robot                = NAMOenv.robot
  leftarm_manip        = robot.GetManipulator('leftarm')
  rightarm_manip       = robot.GetManipulator('rightarm')
  rightarm_torso_manip = robot.GetManipulator('rightarm_torso')

  places = []; contexts = []; picks = []

  robot_curr_conf = problem['robot_initial_config']
  f_vec = compute_fetch_vec(key_configs,problem['original_path'],robot,NAMOenv.env)

  problem['env'].SetViewer('qtcoin')


  for i,node_idx in enumerate(traj):
    print 'traj progress %d/%d'%(i,len(traj))
    node = nodes[node_idx]
    if (node.sample is None): continue

    if not 'isGoal' in dir(node):
      node.isGoal = False

    obj_name  = node.sample['obj']  #curr obj
    assert( obj_name == NAMOenv.curr_obj_name)
    curr_obj  = NAMOenv.env.GetKinBody(obj_name)
    obj_pose  = get_body_xytheta(curr_obj)
    obj_shape = problem['obj_shapes'][obj_name]   

    is_place_node  = ('place_base_pose' in node.sample.keys())
    if is_place_node:
      abs_place_base = np.array(node.sample['place_base_pose'])[None,:]
      clean_pose_data( abs_place_base )
      action = abs_place_base
      if node.visited:  
        NAMOenv.apply_place_action(action)
        robot_curr_conf = abs_place_base
        assert( np.all(np.isclose(robot_curr_conf,get_body_xytheta(robot))) )
        NAMOenv.collided_objs = NAMOenv.compute_obj_collisions()
        NAMOenv.curr_obj_name = NAMOenv.collided_objs[0] 
        continue
      occ_vec = compute_occ_vec( key_configs, robot, NAMOenv.env )
      assert( np.all(np.isclose(robot_curr_conf,get_body_xytheta(robot))) )
      state   = {'c0':robot_curr_conf,\
                 'o_pose':obj_pose,\
                 'o_shape':obj_shape,\
                 'c_vec':occ_vec,\
                 'f_vec':f_vec,\
                 'target_o_pose':target_obj_pose} 
      is_succ = NAMOenv.apply_place_action( action )
      if not is_succ:
        import pdb;pdb.set_trace()
        break
      robot_curr_conf = abs_place_base
      before = len(NAMOenv.collided_objs)
      NAMOenv.collided_objs = NAMOenv.compute_obj_collisions()
      after = len(NAMOenv.collided_objs)
      node.reward = 1 if before-after == 1 else -1
      if len(NAMOenv.collided_objs)==0:
        node.reward = 10
        node.isGoal = True
        n=node.parent
        node.visited=True
        node.action  = action
        node.context = state
        node.score   = 'dummy'
        node.visited = True
        while n is not None:
          n.isGoal = True
          n=n.parent  
        break
      NAMOenv.curr_obj_name = NAMOenv.collided_objs[0] 
    else:
      abs_pick_base = node.sample['pick_base_pose'][None,:] # This is in abs 
      clean_pose_data(abs_pick_base)
      curr_obj_xy = get_body_xytheta(curr_obj)[0,0:2]

      rel_xy        = np.zeros((1,2))
      rel_xy[0,0:2] = abs_pick_base[0,0:2] - curr_obj_xy[0:2]
      arm_len = 0.9844 # determined by spreading out the arm and measuring the dist from shoulder 
      dist_to_grasp = np.linalg.norm(rel_xy)
      assert(dist_to_grasp < arm_len)
 
      th_to_be_set  = compute_angle_to_be_set(curr_obj_xy,\
                                              abs_pick_base[0,0:2])
      executed_th   = abs_pick_base[0,-1]
      rel_th        = executed_th-th_to_be_set
      while abs(rel_th) > np.pi/6.0:
        rel_th = executed_th-2*np.pi-th_to_be_set
      rel_pick_base    = np.hstack([rel_xy,[[rel_th]]]).squeeze()

      # convert it back to abs_pick_base for testing purpose
      pick_base_action = convert_rel_to_abs_base_pose(rel_pick_base,\
                                                      curr_obj_xy) 
      clean_pose_data(pick_base_action)
      assert( np.all( np.isclose(pick_base_action,abs_pick_base) ) )

      action = np.c_[np.array([node.sample['grasp_params']]),\
                     rel_pick_base[None,:]]
      if node.visited:  
        NAMOenv.apply_pick_action(action)
        robot_curr_conf = abs_pick_base
        assert( np.all(np.isclose(robot_curr_conf,get_body_xytheta(robot) )) )
        continue
      occ_vec   = compute_occ_vec( key_configs, robot, NAMOenv.env)
      assert( np.all(np.isclose(robot_curr_conf,get_body_xytheta(robot))) )
      state = {'c0':robot_curr_conf,\
               'o_pose':obj_pose,\
               'o_shape':obj_shape,\
               'c_vec':occ_vec,\
               'f_vec':f_vec,\
               'target_o_pose':target_obj_pose} 
      is_succ = NAMOenv.apply_pick_action( action )
      if not is_succ:
        print "Pick Failed" 
        import pdb;pdb.set_trace()
        sys.exit(-1)
      robot_curr_conf = node.sample['pick_base_pose']
      node.reward  = 0

    node.action  = action
    node.context = state
    node.score   = 'dummy'
    node.visited = True

def get_trajectories(node_idx,nodes,traj,traj_list):
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
    child_traj,_ = get_trajectories(child_idx,nodes,new_traj,traj_list)

  if len(node.children) == 0:
    traj_list.append(traj)
  
  return traj,traj_list

def create_RL_data(nodes,key_configs,NAMOenv):
  # initialize nodes
  for n in nodes:   
    n.visited = False
    if (n.sample is not None) and isinstance(n.sample,dict)\
       and 'path' in n.sample.keys(): 
      n.sample.pop('path',None) # remove path; too large

  root_node_idxs = [get_node_idx(n,nodes) for n in nodes if n.sample == None]
  for root_node_idx in root_node_idxs:
    _,traj_list = get_trajectories( node_idx=root_node_idx,\
                                    nodes=nodes, traj=[], traj_list=[]) 
    for idx,traj in enumerate(traj_list):
      print 'processed trajs %d/%d'%(idx,len(traj_list))
      get_training_data_from_traj(traj,nodes,key_configs,NAMOenv)
  
  RL_data = []
  for traj in traj_list:
    A=[];S=[];R=[];IsGoalTraj=[]
    for step in traj:
      node=nodes[step]
      if (node.sample is None): continue
      if not node.visited: 
        import pdb;pdb.set_trace()
        raise ValueError
      # How should I represent a state?
      # For placement
      #   fetch path
      #   collision vector
      #   robot pose
      # For picking
      #   fetch path
      #   collision vector
      #   object pose
      #   robot pose
      # I am thinking fetch vector, and 
      A.append(node.action)
      S.append(node.context)
      R.append(node.reward)
      IsGoalTraj.append(node.isGoal)
    RL_data.append( {'s':S,'r':R,'a':A,'G':IsGoalTraj} )
  return RL_data

def setup_env(env_file):
  NAMOenv = NAMO(env_file,is_preprocess=True)
  return NAMOenv
 
def process_search_episode_file( raw_file,raw_dir,key_configs,processed_dir ):
  env_num = raw_file.split('.pkl')[0]
  if os.path.isfile(processed_dir+'/RL_train_data_'+str(env_num)+'.p'): 
    print 'already have '+processed_dir+'/RL_train_data_'+str(env_num)+'.p'
  
    #return
  
  #search_episode_data = pickle.load(open(raw_dir+raw_file,'r'))
  #env_file            = pickle.load(open(raw_dir+raw_file,'r'))
  manual_pinst = raw_dir+raw_file
  NAMOenv      = setup_env(manual_pinst)
  #NAMOenv.env.SetViewer('qtcoin')

  nodes = pickle.load(open(raw_dir+raw_file,'r'))['nodes']
  trajs = create_RL_data(nodes,key_configs,NAMOenv)
  pickle.dump(trajs,\
              open(processed_dir+'/RL_train_data_'+str(env_num)+'.p','wb'))
  RaveDestroy()
  NAMOenv.env.Destroy()

def main():
  key_configs   = pickle.load(open('./key_configs/key_configs.p','r'))
  processed_dir = './processed_train_data/'
  raw_dir       = './raw_train_data/'
  raw_dir       = '../../AdversarialOptimization/NAMO/raw_train_data/'
  #raw_dir = '/data/public/rw/pass.port/NAMO/raw_train_data/'
  raw_file_list = os.listdir(raw_dir)

  if len(sys.argv) == 1:
    # serial case
    for raw_file in raw_file_list[2:]:
      if raw_file.find('.pkl') == -1:continue
      if raw_file.find('1114') != -1: continue # but why does this happen
      print raw_file
      process_search_episode_file(raw_file,raw_dir,key_configs,processed_dir )
  else:
    # multithreads case
    raw_file_idx = int(sys.argv[1])
    raw_file = raw_file_list[raw_file_idx]
    if raw_file.find('1114') != -1: 
      return
    if not os.path.isfile(raw_dir+raw_file):
      print 'do not have the raw file for this idx'
    process_search_episode_file( raw_file,raw_dir,key_configs,processed_dir )

if __name__ == '__main__':
  main()
 


