import os
import pickle
import copy
import numpy as np
import pickle
import time
import sys

sys.path.append('../mover_library/')
from samplers import set_robot_config
from NAMO_problem import NAMO_problem,pick_obj,place_obj
from openravepy import *
from manipulation.primitives.transforms import get_point,set_point,set_quat,quat_from_angle_vector,get_quat
from manipulation.bodies.bodies import set_config
from manipulation.bodies.bodies import box_body, randomly_place_body, place_xyz_body
from manipulation.motion.primitives import extend_fn,distance_fn,sample_fn,collision_fn

from preprocessing_utils import *
from utils import compute_occ_vec
from preprocessing_utils import compute_fetch_vec

def aggregate_data( processed_dir ): 
  plans = []
  env_idxs=[]
  # remember that I need to revive the environment
  for fin in os.listdir( processed_dir ):
    if fin.find('box') == -1: continue  
    if fin.find('box_aggregated') != -1: continue  
    env_idx = int( fin.split('box_train_data_')[1].split('episode_')[1].split('.p')[0] )
    plans.append ( pickle.load(open(processed_dir+fin,'r'))[0] )
    env_idxs.append( env_idx )  
    if len(plans) > 282:
      break
  pickle.dump( plans,open('./processed_train_data/box_aggregated_data.py','wb'))

def get_training_data_from_traj(traj,nodes):
  # check if traj is a successful node
  n_to_move = len(nodes[0].state)
  n_remain   = len(nodes[traj[-1]].state)
  score=n_to_move
  
  # if not, return None
  if n_remain > 0:
    return None,None

  # extract action data 
  plan = []
  for node_idx in traj:
    node = nodes[node_idx]
    if node.sample == None: continue
    plan.append( node.sample )

  return plan,score

def process_search_episode(data):
  nodes = data['nodes']
  for n in nodes:   
    n.visited = False
    if (n.sample is not None) and isinstance(n.sample,dict)\
       and 'path' in n.sample.keys(): 
      n.sample.pop('path',None) # remove path; too large

  # get all the trajectories encountered during search
  stime = time.time()

  plan_data=[]
  score_data=[]

  root_node_idxs = [get_node_idx(n,nodes) for n in nodes if n.sample == None]
  for root_node_idx in root_node_idxs:
    _,traj_list = get_trajectories( node_idx=root_node_idx,nodes=nodes, traj=[], traj_list=[]) 
    
    for traj in traj_list:
      plan,score =  get_training_data_from_traj(traj,nodes)
      if plan is not None:
        plan_data.append(plan)
        score_data.append(score)
  return plan_data,score_data

def process_search_episode_file( raw_file,raw_dir,processed_dir ):
  env_num = raw_file.split('.pkl')[0]
  f_processed = processed_dir+'/box_train_data_'+str(env_num)+'.p'
  if os.path.isfile(f_processed): 
    print 'already have'
    return
  
  search_episode_data = pickle.load(open(raw_dir+raw_file,'r'))
  plan_data,score_data = process_search_episode(search_episode_data)

  if len(plan_data)>0:
    data = [plan_data,score_data]
    print 'Saving',f_processed,len(plan_data),score_data
    pickle.dump(data,open(f_processed,'wb'))

def preprocess_data():
  raw_dir       = './train_data/'
  raw_file_list = [f for f in os.listdir(raw_dir) if f.find('.pkl')!=-1]
  processed_dir = './processed_train_data/'

  for raw_file in raw_file_list:
    process_search_episode_file( raw_file,raw_dir,processed_dir )

def main():
  processed_dir = './processed_train_data/'
  aggregate_data(processed_dir)
    
if __name__ == '__main__':
  main()
