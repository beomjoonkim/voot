import os
import copy
import numpy as np
import time
import sys
import pickle

sys.path.append('../mover_library/')
from utils import set_robot_config
from NAMO_problem import NAMO_problem
from NAMO_problem import NAMO_problem,pick_obj,place_obj

from openravepy import *
from manipulation.primitives.transforms import get_point,set_point,set_quat,quat_from_angle_vector
from manipulation.bodies.bodies import set_config
from manipulation.bodies.bodies import box_body, randomly_place_body, place_xyz_body
from manipulation.motion.primitives import extend_fn,distance_fn,sample_fn,collision_fn
from preprocessing_utils import compute_fetch_vec
from key_config_utils import c_outside_threshold
from utils import compute_occ_vec,clean_pose_data


def node_made_progress_in_this_traj(nodes,node_idx,traj):
  n_collided_parent = len( nodes[node_idx].parent.state )
  n_new_collided    = len( nodes[node_idx].state ) 

  traj_nidx_idx = np.where(traj==node_idx)[0][0]
  
  grand_child_idx = traj_nidx_idx = traj_nidx_idx+2
  has_grand_child = grand_child_idx<len(traj)
  is_last_node_in_traj = traj_nidx_idx == len(traj)
  if is_last_node_in_traj:
    n_child_collided = n_new_collided
  elif has_grand_child:
    is_child_is_pick_node = 'g_config' in nodes[traj[traj_nidx_idx+1]].sample.keys())
    assert( is_child_pick_node ) 
    next_node = nodes[traj[grand_child_idx]]
    n_child_collided  = len( next_node.state )
  else:
    n_child_collided=n_new_collided

  if (n_collided_parent == n_new_collided) and (n_new_collided==n_child_collided):
    return False

  return True
"""
def compute_fetch_vec( key_configs, fetch_path, robot, env):
  fetch_vec = [[]]*len(key_configs)
  xy_threshold = 0.3 # size of the base - 0.16
  th_threshold = 20*np.pi/180  # adhoc
  fetch_path = fetch_path[::int(0.1*len(fetch_path))]
  for f in fetch_path:
    for kidx,k in enumerate(key_configs):   
      xy_dist = np.linalg.norm(f[0:2]-k[0:2])
      th_dist = abs(f[2]-k[2]) if abs(f[2]-k[2])<np.pi else 2*np.pi-abs(f[2]-k[2])
      if xy_dist < xy_threshold and th_dist < th_threshold:
        fetch_vec[kidx] = 1
      else:
        fetch_vec[kidx] = 0
  return fetch_vec
"""   
def get_training_data_from_traj(traj,nodes,key_configs,problem):
  actions=[]
  scores = []
  contexts = []

  # total number of objs moved? n objs at the beginning - n objs at the end
  score = len(problem['collided_objs']) - len(nodes[traj[-1]].state) 

  obj_shapes = problem['obj_shapes']
  obj_poses = problem['obj_poses']
  env = problem['env']
  problem['initial_state'][0].Restore()
  
  robot = env.GetRobots()[0]
  leftarm_manip = robot.GetManipulator('leftarm')
  rightarm_manip = robot.GetManipulator('rightarm')
  rightarm_torso_manip = robot.GetManipulator('rightarm_torso')

  target_obj_name = problem['target_obj']
  target_obj_pose = obj_poses[target_obj_name]
  target_obj_shape = obj_shapes[target_obj_name]

  for node_idx in traj:
    node = nodes[node_idx]
    is_root_node = node.sample is None
    if is_root_node: continue

    # place the object if it was previously visited
    is_place_node  = ('place_base_pose' in node.sample.keys())
    if node.visited and is_place_node:  
      set_robot_config( node.sample['place_base_pose'],robot )
      place_obj( robot.GetGrabbed()[0],robot,leftarm_manip,rightarm_manip )
      if node.score < score:
        node.score = score
      continue

    # pick the obj if it was a pick
    if not is_place_node:  #'place_base_pose' not in node.sample.keys(): 
      set_robot_config( node.sample['pick_base_pose'],robot )
      robot_curr_conf = node.sample['pick_base_pose'] 
      obj = env.GetKinBody(node.state[0]) 
      pick_obj( obj, robot, node.sample['g_config'], \
                leftarm_manip,rightarm_torso_manip)
      continue

    if node_made_progress_in_this_traj(nodes,node_idx,traj):
      score = len(problem['collided_objs']) - len(nodes[traj[-1]].state) 
    else: 
      score = 0
    print score

    # for place, curr obj is the obj that it just placed, and 
    # state is the state achieved by placing it.
    # Or, it can jsut be the one htat is grabbed
    obj = robot.GetGrabbed()[0]
    obj_name = obj.GetName()
    assert( obj_name == node.parent.state[0] ) # a little t est
    assert( obj_name != target_obj_name)

    # proposed location for base pose
    abs_place_base = np.array(node.sample['place_base_pose'])
    action = np.array([abs_place_base] )
    clean_pose_data(action)

    stime = time.time()
    occ_vec = compute_occ_vec( key_configs, robot, env)
    print time.time()-stime

    stime = time.time()
    f_vec = compute_fetch_vec( key_configs, problem['original_path'], robot, env)
    print time.time()-stime

    set_robot_config( abs_place_base,robot )
    place_obj( robot.GetGrabbed()[0],robot,leftarm_manip,rightarm_manip )

    assert(np.all( robot_curr_conf == node.parent.sample['pick_base_pose']))
    
    # add each data
    context = {'c0':robot_curr_conf,\
               'c_vec':occ_vec,\
               'target_o_pose':target_obj_pose,\
               'f_vec':f_vec}
    node.action  = action
    node.context = context
    node.score = score
    node.visited = True

def main():
  key_configs   = pickle.load(open('./key_configs.p','r'))
  processed_dir = './processed_train_data/'
  raw_dir       = './train_data/'
  raw_file_list = os.listdir(raw_dir)

  if len(sys.argv) == 1:
    # serial case
    for raw_file in raw_file_list:
      print 'raw_file:',raw_file
      if raw_file.find('.pkl') == -1:
        continue
      env_num = raw_file.split('.pkl')[0]
      processed_fname = processed_dir+'/place_train_data_'+str(env_num)+'.p'
      if os.path.isfile( processed_fname ): 
        print 'already have'
        continue
      process_search_episode_file( raw_file,raw_dir,key_configs,\
                                      processed_fname, get_training_data_from_traj)
    n_data = aggregate_data( processed_dir, 'place' )
    agg_data = pickle.load(open(processed_dir+'aggregated_data_place_n_'\
                           +str(n_data)+'.p','r'))
    print 'n_data is', np.array(n_data).shape
    test_get_training_data_from_traj()
  else:
    # parallel case
    raw_file_idx = int(sys.argv[1])
    raw_file = raw_file_list[raw_file_idx]
    if not os.path.isfile(raw_dir+raw_file):
      print 'do not have the raw file for this idx'
      return
    env_num = raw_file.split('.pkl')[0]
    processed_fname = processed_dir+'/place_train_data_'+str(env_num)+'.p'
    if os.path.isfile( processed_fname ):
      print 'exists'
      #return
    process_search_episode_file( raw_file,raw_dir,key_configs,processed_fname,\
                                    get_training_data_from_traj )

if __name__ == '__main__':
  main()

