import cPickle as pickle

import os
import numpy as np
from openravepy import *
from conveyor_belt_problem import two_tables_through_door
import copy



def setup_env(env_file):
  try:
    obj_shapes  = env_file['obj_shapes']
    obj_poses   = env_file['obj_poses']
    obst_shapes = env_file['obst_shapes'] 
    obst_poses  = env_file['obst_poses'] 
  except KeyError:
    return

  env = Environment()
  problem = two_tables_through_door(env,obj_shapes,obj_poses,obst_shapes,obst_poses)

  return problem

def get_node_idx(ntarget,nodes):
  return np.where([ ntarget==n for n in nodes])[0][0]

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

 
