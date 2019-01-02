import pickle
import os
import copy

import numpy as np
from openravepy import *

def compute_fetch_vec( key_configs, fetch_path, robot, env):
  # what is this vector?
  # it is a fetch path represented with key configuration obstacles
  # if a fetch path configuration is within the specified threshold wrt key configs, then 
  # turn on the bit
  fetch_vec = [0]*len(key_configs)
  xy_threshold = 0.3 # size of the base - 0.16
  th_threshold = 20*np.pi/180  # adhoc
  fetch_path_n = max(int(0.01*len(fetch_path)),1)
  fetch_path = fetch_path[::fetch_path_n]
  for kidx,k in enumerate(key_configs):   
    for f in fetch_path:
      xy_dist = np.linalg.norm(f[0:2]-k[0:2])
      th_dist = abs(f[2]-k[2]) if abs(f[2]-k[2])<np.pi else 2*np.pi-abs(f[2]-k[2])
      if xy_dist < xy_threshold and th_dist < th_threshold:
        fetch_vec[kidx] = 1
  return fetch_vec

def process_search_episode( data, get_training_data_from_traj,problem,key_configs ):
  obj_poses = data['obj_poses']
  obj_shapes = data['obj_shapes']
  nodes = data['nodes']
  for n in nodes: 
    n.visited = False
    if (n.sample is not None) and isinstance(n.sample,dict)\
       and 'path' in n.sample.keys(): 
      n.sample.pop('path',None) # remove path; too large
  if len(nodes)==0: return None,None,None

  # get all the trajectories encountered during search
  x_data         = []
  score_data   = []
  context_data = []
  
  root_node_idxs = [get_node_idx(n,nodes) for n in nodes if n.sample == None]
  for root_node_idx in root_node_idxs:
    _,traj_list = get_trajectories( node_idx=root_node_idx,nodes=nodes, traj=[], traj_list=[]) 
    for traj in traj_list:
      get_training_data_from_traj(traj,nodes,key_configs,problem)

  for node in nodes:
    if (node.sample is None): continue
    if not node.visited: continue
    x_data.append(node.action)
    score_data.append(node.score)
    context_data.append(node.context)

  return x_data,score_data,context_data

def aggregate_data( train_dir,fname_keyword ):
  # loops through train files in train dir, loading data from files with fname_keyword in it
  train_dir = train_dir+'/'
  for fdata in os.listdir(train_dir):
    if fdata.find(fname_keyword) == -1: continue
    if fdata.find('aggregated') != -1: continue
    print fdata
    data=pickle.load(open( train_dir + fdata))
    if 'x_data' not in locals():
      x_data = np.array(data[0])
      s_data = np.array(data[1])[:,None]
      c_data = np.array(data[2])
    else:
      try:
        x_data = np.vstack((x_data,np.array(data[0])))
        s_data = np.vstack((s_data,np.array(data[1])[:,None]))
        c_data = np.vstack((c_data,np.array(data[2])))
      except:
        print 'rm ' +train_dir+fdata
        os.system('rm '+ train_dir+fdata)
    data = [x_data,s_data,c_data]
    if len(x_data) > 5000:
      print 'saving',len(x_data)
      with open(train_dir+fname_keyword+'_aggregated_data.p','wb') as foutput:
        pickle.dump(data,foutput)
      return len(x_data)

def get_trajectories(node_idx,nodes,traj,traj_list):
  # inputs: root node idx and nodes, empty traj and traj list
  # outputs:
  #   new_traj: a list of a seuqence from a root node to a leaf node
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

def get_node_idx(ntarget,nodes):
  return np.where([ ntarget==n for n in nodes])[0][0]

def process_search_episode_file( raw_file,raw_dir,key_configs,processed_fpath,\
                                 get_training_data_from_traj ):
  # loads raw file, setup environment, and calls process search episode
  
  search_episode_data = pickle.load(open(raw_dir+raw_file,'r'))
  env_file = pickle.load(open(raw_dir+raw_file,'r'))
  problem = setup_env(env_file)
  actions,scores,contexts = process_search_episode(search_episode_data,\
                                                   get_training_data_from_traj,
                                                   problem,key_configs)
  RaveDestroy()
  problem['env'].Destroy()
  pickle.dump([actions,scores,contexts],\
            open(processed_fpath,'wb'))


