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

#from preprocess_place_data import get_trajectories,get_node_idx

def node_made_progress_in_this_traj(nodes,node_idx,traj):
  n_collided    = len( nodes[node_idx].state ) 

  traj_nidx_idx = np.where(traj==node_idx)[0][0]
  child_idx = traj_nidx_idx = traj_nidx_idx+1
  has_child = child_idx<len(traj)

  if has_child:
    n_child_collided = len(nodes[traj[child_idx]].state)
  else:
    n_child_collided = n_collided

  will_clear_obj = not(n_collided==n_child_collided)
  if will_clear_obj:
    return True
  else:
    return False

def get_training_data_from_traj(traj,nodes,key_configs,problem):
  actions=[]
  scores = []
  contexts = []

  # total number of objs moved? n objs at the beginning - n objs at the end
  score = len(problem['collided_objs']) - len(nodes[traj[-1]].state) 

  # score = number of objects to be moved
  obj_shapes = problem['obj_shapes']
  obj_poses = problem['obj_poses']
  env = problem['env']
  problem['initial_state'][0].Restore()
  
  robot = env.GetRobots()[0]
  leftarm_manip = robot.GetManipulator('leftarm')
  rightarm_manip = robot.GetManipulator('rightarm')
  rightarm_torso_manip = robot.GetManipulator('rightarm_torso')
  #env.SetViewer('qtcoin')

  target_obj_name = problem['target_obj']
  target_obj_pose = obj_poses[target_obj_name]
  robot_curr_conf = problem['robot_initial_config']
  for node_idx in traj:
    node = nodes[node_idx]
    if (node.sample is None): continue
  
    is_place_node  = ('place_base_pose' in node.sample.keys())
    if is_place_node: 
      set_robot_config( node.sample['place_base_pose'],robot )
      robot_curr_conf = node.sample['place_base_pose'] 
      place_obj( robot.GetGrabbed()[0],robot,leftarm_manip,rightarm_manip )
      continue

    # curr obj is the first obj of the state; checked with line 193 of mover_problem.py
    obj_name  = node.state[0] 
    obj_pose  = obj_poses[obj_name] 
    obj_shape = obj_shapes[obj_name]
    obj       = env.GetKinBody(obj_name)

    # pick the object if it was previously visited
    if node.visited and (not is_place_node):  
      set_robot_config( node.sample['pick_base_pose'],robot )
      pick_obj( obj, robot, node.sample['g_config'], \
                leftarm_manip,rightarm_torso_manip)
      if node.score < score:
        node.score = score
      continue

    if node_made_progress_in_this_traj(nodes,node_idx,traj):
      score = len(problem['collided_objs']) - len(nodes[traj[-1]].state) 
    else: 
      score = 0
    print score

    # What happens if we move the same object twice?
    # How to get Opose?
    #th = abs_pick_base[-1] # should this be relative as well? No.
    #rel_pick_base =np.append( abs_pick_base[0:2] - obj_pose[0:2], th )   
    abs_pick_base = node.sample['pick_base_pose']
    if abs_pick_base[-1] < 0: abs_pick_base[-1]+=2*np.pi
    assert(abs_pick_base[-1]>=0 and abs_pick_base[-1]<=2*np.pi)

    # make sure obj is where it should be
    try:
      assert( np.all(np.isclose( get_point(obj)[0:2] ,obj_pose[0:2] ) ) )
    except: 
      #NOTE: when you go and test pick soap, 
      # make sure you apply the following to get th of obj
      Tbefore = obj.GetTransform()
      obj_quat = get_quat(obj)
      th1 = np.arccos(obj_quat[0])*2
      th2 = np.arccos(-obj_quat[0])*2
      th3 = -np.arccos(obj_quat[0])*2
      quat_th1=quat_from_angle_vector(th1,np.array([0,0,1]))
      quat_th2=quat_from_angle_vector(th2,np.array([0,0,1]))
      quat_th3=quat_from_angle_vector(th3,np.array([0,0,1]))
      if np.all( np.isclose(obj_quat,quat_th1)):
        th=th1
      elif np.all( np.isclose(obj_quat,quat_th2)):
        th=th2  
      elif np.all(np.isclose(obj_quat,quat_th3)):
        th=th3
      else:
        print "This should not happen"
        import pdb;pdb.set_trace()
      if th<0: th+=2*np.pi
      assert(th>=0 and th<2*np.pi)
      set_quat(robot,quat_from_angle_vector(th,np.array([0,0,1])))
      Tafter = obj.GetTransform()
      assert(np.all(np.isclose(Tbefore,Tafter)))
      obj_pose = np.hstack([get_point(obj)[0:2],th])

    action = np.concatenate([node.sample['grasp_params'],abs_pick_base] )
    stime = time.time()
    occ_vec = compute_occ_vec( key_configs, robot, env)
    print time.time()-stime

    stime = time.time()
    f_vec = compute_fetch_vec( key_configs, problem['original_path'], robot, env)
    print time.time()-stime
    
    # pick the object
    set_robot_config( node.sample['pick_base_pose'],robot )
    pick_obj( obj, robot, node.sample['g_config'], \
              leftarm_manip,rightarm_torso_manip)

    # add each data
    context = {'c0':robot_curr_conf,\
               'c_vec':occ_vec,\
               'o_pose':obj_pose,\
               'o_shape':obj_shape,\
               'target_o_pose':target_obj_pose,\
               'f_vec':f_vec}
    node.action  = action
    node.context = context
    node.score = score
    node.visited = True

def test_get_training_data_from_traj():
  #TODO
  # what to test?
  # score actually correct - take a traj, and check the score of each node
  # action: context's obj_pose + action['pick_base_pose'] = node.sample['pick_base_pose']
  raw_dir       = './train_data/'
  raw_file_list = os.listdir(raw_dir)
  raw_file = raw_file_list[0]

  env_num = 595
  raw_data = pickle.load(open(raw_dir+'episode_'+str(env_num)+'.pkl','r'))
  proc_data = pickle.load( open('./processed_train_data/pick_train_data_episode_595.p','r'))

  nodes = raw_data['nodes']
  root_node_idxs = [get_node_idx(n,nodes) for n in nodes if n.sample == None]
  
  for root_node_idx in root_node_idxs:
    _,traj_list = get_trajectories( node_idx=root_node_idx,nodes=nodes, traj=[], traj_list=[]) 

  n_total_objs = len(nodes[root_node_idxs[0]].state)
  orig_robot_conf = nodes[root_node_idxs[0]]
  
  # load the saved data
  actions  = proc_data[0]
  scores   = proc_data[1]
  contexts = proc_data[2]
  for a_,s_,c_ in zip(actions,scores,contexts):
    grasp_params    = a_[:-3]
    # find the corresponding node
    node = [n for n in nodes if n.sample is not None\
                               and 'grasp_params' in n.sample.keys()\
                               and np.all(n.sample['grasp_params']==grasp_params)][0]
    trajs_with_node = [ traj for traj in traj_list for nidx in traj if node == nodes[nidx] ]
    rel_pick_base   = a_[-3:]
    
    robot_curr_conf = c_[0]
    obj_pose        = c_[1]
    obj_shape       = c_[2]
    occ_vec         = c_[3]
  
    # absolute pose check
    abs_pick_pose = rel_pick_base
    abs_pick_pose[0:2] = abs_pick_pose[0:2]+obj_pose[0:2]
    n_abs_pick_base = node.sample['pick_base_pose']
    th = n_abs_pick_base[-1]; 
    if th<0: th+=np.pi*2
    try: 
      assert( np.all(abs_pick_pose[0:2] == n_abs_pick_base[0:2]) )
      assert( th == abs_pick_pose[-1] )
    except AssertionError:
      print 'abs pick pose err'
      import pdb;pdb.set_trace()

    # score check
    true_score = max( [n_total_objs - len(nodes[traj[-1]].state) for traj in trajs_with_node] )
    try: 
      assert( s_ == true_score)
    except AssertionError:
      print 'score err'
      import pdb;pdb.set_trace()

    # robot conf where it is supposed to be
    if node.parent.sample is not None:
      try:
        assert( np.all(robot_curr_conf == node.parent.sample['place_base_pose']) )
      except AssertionError:
        print 'curr conf err'
        import pdb;pdb.set_trace()
    else:
      try:
        assert( np.all(robot_curr_conf == np.array([-1,  1,  0])) )
      except AssertionError:
        print 'curr conf err'
        import pdb;pdb.set_trace()

    # objs where they are is supposed to be
    #TODO
    print "All tests passed"


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
      processed_fname = processed_dir+'/pick_train_data_'+str(env_num)+'.p'
      if os.path.isfile( processed_fname ): 
        print 'already have'
        continue
      process_search_episode_file( raw_file,raw_dir,key_configs,\
                                    processed_fname, get_training_data_from_traj)
    test_get_training_data_from_traj()
  else:
    # parallel case
    raw_file_idx = int(sys.argv[1])
    raw_file = raw_file_list[raw_file_idx]
    if not os.path.isfile(raw_dir+raw_file):
      print 'do not have the raw file for this idx'
      return
    env_num = raw_file.split('.pkl')[0]
    processed_fname = processed_dir+'/pick_train_data_'+str(env_num)+'.p'
    if os.path.isfile( processed_fname ):
      print 'exists'
      return
    process_search_episode_file( raw_file,raw_dir,key_configs,processed_fname,\
                                    get_training_data_from_traj )

if __name__ == '__main__':
  main()

