import os
import pickle
import copy
import numpy as np
import pickle
import time
import sys

from openravepy import *
from conveyor_belt_problem import two_tables_through_door
from manipulation.primitives.transforms import get_point,set_point,set_quat,quat_from_angle_vector
from manipulation.bodies.bodies import set_config
from manipulation.bodies.bodies import box_body, randomly_place_body, place_xyz_body
from manipulation.motion.primitives import extend_fn,distance_fn,sample_fn,collision_fn

sys.path.append('../mover_library/')
from samplers import set_robot_config
from preprocessing_utils import setup_env,get_trajectories,get_node_idx


train_data_dir = './train_data/'


def get_training_data_from_traj(traj,nodes,key_configs,problem):
  env = problem['env']
  init_state = problem['initial_state'][0]
  robot = env.GetRobots()[0]
  leftarm_manip = robot.GetManipulator('leftarm')
  rightarm_manip = robot.GetManipulator('rightarm')
  rightarm_torso_manip = robot.GetManipulator('rightarm_torso')

  places = []
  contexts = []
  scores = []
  score = len(nodes[traj[-1]].state)
  for node_idx in traj:
    node = nodes[node_idx]
    init_state.Restore()
    if (node.sample is None): continue
    if 'place_base_pose' not in node.sample.keys(): continue # if not place, continue
    if node.visited: 
      if node.score < score:
        node.score = score
      continue
    obj_name  = node.sample['obj'] 
    obj_pose  = problem['obj_poses'][obj_name]
    obj_shape = problem['obj_shapes'][obj_name]   
    obj_idx   = int(obj_name.split('obj')[1])
    curr_obj = env.GetKinBody(obj_name)

    # get robot into its grasp form - how to get this? Get the parent
    pick_node = node.parent  
    base_pose = pick_node.sample['pick_base_pose']
    g_config  = pick_node.sample['g_config']

    # pick up the object
    set_robot_config(base_pose,robot)
    set_config(robot,g_config[0],leftarm_manip.GetArmIndices())
    set_config(robot,g_config[1],rightarm_torso_manip.GetArmIndices())
    robot.Grab(curr_obj)  

    # go through all the placements in the parents
    temp_curr_n = node.parent
    while temp_curr_n.sample is not None:
      temp_curr_obj = env.GetKinBody( temp_curr_n.sample['obj'] )
      if 'place_base_pose' in temp_curr_n.sample:
        # place obj
        obj_xytheta = temp_curr_n.sample['place_obj_pose']
        x  = temp_curr_n.sample['place_obj_pose'][0]
        y  = temp_curr_n.sample['place_obj_pose'][1]
        z  = get_point(temp_curr_obj)[-1]
        th = temp_curr_n.sample['place_obj_pose'][2]
        set_point(temp_curr_obj,[x,y,z+0.000001])
        th = quat_from_angle_vector(th,np.array([0,0,1]))
        set_quat(temp_curr_obj,th)
      temp_curr_n = temp_curr_n.parent

    # I am holding the object
    stime=time.time()
    occ_vec = []
    for config in key_configs:
      set_robot_config(config,robot)
      occ_vec.append( env.CheckCollision(robot) )
    context = np.array(occ_vec)
    print time.time()-stime
    action = np.concatenate([node.sample['place_obj_pose'],node.sample['place_base_pose']])   

    node.action  = action
    node.context = context
    node.obj_idx = obj_idx
    node.score   = score
    node.reward  = len(node.state)-len(node.parent.state) # Reward at this state
    node.visited = True
  
def process_search_episode(data,key_configs,problem):
  nodes = data['nodes']
  for n in nodes:   
    n.visited = False
    if (n.sample is not None) and isinstance(n.sample,dict)\
       and 'path' in n.sample.keys(): 
      n.sample.pop('path',None) # remove path; too large

  # get all the trajectories encountered during search
  stime = time.time()

  place_data=[]
  score_data=[]
  context_data=[]
  obj_idx_data= []

  root_node_idxs = [get_node_idx(n,nodes) for n in nodes if n.sample == None]
  for root_node_idx in root_node_idxs:
    _,traj_list = get_trajectories( node_idx=root_node_idx,nodes=nodes, traj=[], traj_list=[]) 
    for traj in traj_list:
      get_training_data_from_traj(traj,nodes,key_configs,problem)
  
  for node in nodes:
    if (node.sample is None): continue
    if 'place_base_pose' not in node.sample.keys(): continue
    if not node.visited: raise ValueError
    place_data.append(node.action)
    score_data.append(node.score)
    context_data.append(node.context)
    obj_idx_data.append(node.obj_idx)
  return place_data,score_data,context_data,obj_idx_data

def create_RL_data(data,key_configs,problem):
  nodes = data['nodes']
  # initialize nodes
  for n in nodes:   
    n.visited = False
    if (n.sample is not None) and isinstance(n.sample,dict)\
       and 'path' in n.sample.keys(): 
      n.sample.pop('path',None) # remove path; too large

  root_node_idxs = [get_node_idx(n,nodes) for n in nodes if n.sample == None]
  for root_node_idx in root_node_idxs:
    _,traj_list = get_trajectories( node_idx=root_node_idx,nodes=nodes, traj=[], traj_list=[]) 
    for traj in traj_list:
      get_training_data_from_traj(traj,nodes,key_configs,problem)
  
  RL_data = []
  for traj in traj_list:
    A=[];S=[];R=[]
    for step in traj:
      node=nodes[step]
      if (node.sample is None): continue
      if 'place_base_pose' not in node.sample.keys(): continue
      if not node.visited: raise ValueError
      A.append(node.action)
      S.append(node.context)
      R.append(node.reward)
    RL_data.append( {'s':S,'r':R,'a':A} )
  return RL_data
 
def process_search_episode_file( raw_file,raw_dir,key_configs,processed_dir ):
  env_num = raw_file.split('.pkl')[0]
  if os.path.isfile(processed_dir+'/RL_train_data_'+str(env_num)+'.p'): 
    print 'already have'
    return
  
  search_episode_data = pickle.load(open(raw_dir+raw_file,'r'))
  env_file = pickle.load(open(raw_dir+raw_file,'r'))
  problem = setup_env(env_file)

  CREATE_PLANNING_DATA=False
  if CREATE_PLANNING_DATA:
    #places,scores,contexts,obj_idxs = process_search_episode(search_episode_data,\
    #                                                        key_configs,problem)
    data = [places,\
            scores,\
            contexts,\
            obj_idxs]
    pickle.dump(data,\
              open(processed_dir+'/place_train_data_'+str(env_num)+'.p','wb'))
  else:
    trajs = create_RL_data(search_episode_data,key_configs,problem)
    pickle.dump(trajs,\
              open(processed_dir+'/RL_train_data_'+str(env_num)+'.p','wb'))
  RaveDestroy()
  problem['env'].Destroy()


def test_get_trajectories():
  key_configs   = [] # pickle.load(open('./key_configs.p','r'))
  raw_dir       = './train_data/'
  raw_file_list = os.listdir(raw_dir)
  for raw_file in raw_file_list:
    if raw_file.find('.pkl') == -1: continue
    search_episode_data = pickle.load(open(raw_dir+raw_file,'r'))

    nodes = search_episode_data['nodes']
    root_node_idxs = [get_node_idx(n,nodes) for n in nodes if n.sample == None]
    for root_node_idx in root_node_idxs:
      _,traj_list = get_trajectories( node_idx=root_node_idx,nodes=nodes, traj=[], traj_list=[]) 


    # Test0: all traj should begin at root node

    # Test1: checking existence of the computed traj
    # trace back to the root node and see if the trajectory actually exists
    for traj in traj_list:
      try:
        assert( nodes[traj[0]].parent is None )
      except:
        print raw_file,'failed root node test'
        import pdb;pdb.set_trace()
      curr_n = nodes[traj[-1]]
      reverse_traj= []
      while curr_n.parent != None:
        reverse_traj.append(  get_node_idx(curr_n,nodes) )
        curr_n = curr_n.parent
      reverse_traj.append(get_node_idx(curr_n,nodes) )
      try: 
        assert( reverse_traj[::-1] == traj )
      except:
        print raw_file,'failed'
        import pdb;pdb.set_trace()

    #todo Test2: check if all the trajs are computed
    print raw_file,'passed'
  print "All tests passed"

def test_get_training_data():
  key_configs   = [] # pickle.load(open('./key_configs.p','r'))
  raw_dir       = './train_data/'
  raw_file_list = os.listdir(raw_dir)

  for raw_file in raw_file_list:
    search_episode_data = pickle.load(open(raw_dir+raw_file,'r'))
    nodes = search_episode_data['nodes']
    problem = setup_env(search_episode_data)
    for n in nodes:   
      n.visited = False
      if (n.sample is not None) and isinstance(n.sample,dict)\
         and 'path' in n.sample.keys(): 
        n.sample.pop('path',None) # remove path; too large

    root_node_idxs = [get_node_idx(n,nodes) for n in nodes if n.sample == None]
    for root_node_idx in root_node_idxs:
      _,traj_list = get_trajectories( node_idx=root_node_idx,nodes=nodes, traj=[], traj_list=[]) 
      for traj in traj_list:
        get_training_data_from_traj(traj,nodes,key_configs,problem)

    # Test0: check it is indeed ith obj
    # Test1: the score of the action is correct
    print 'testing',raw_file
    for traj in traj_list:
      # it might be scored 4 in this traj, but there was a traj that scored it 5
      for node_idx in traj:
        node = nodes[node_idx]
        # scores at which node_idx occurred
        scores = [len(nodes[t[-1]].state) for t in traj_list if node_idx in t]
        true_score = max( scores )
        if (node.sample is None): continue
        if 'place_base_pose' not in node.sample.keys(): continue
        if not node.visited: raise ValueError

        obj_idx = node.obj_idx
        if not obj_idx  == int(node.parent.sample['obj'].split('obj')[1]):
          print 'obj idx mistmatch'
          raise AssertionError 
        
        score = node.score
        if not true_score==score:
          print 'score mistmatch'
          raise AssertionError

    RaveDestroy()
    problem['env'].Destroy()
    print raw_file,'passed'

def main():
  key_configs   = pickle.load(open('./key_configs/key_configs.p','r'))
  processed_dir = './processed_train_data/'
  raw_dir       = './train_data/'
  raw_file_list = os.listdir(raw_dir)

  if len(sys.argv) == 1:
    # serial case
    for raw_file in raw_file_list:
      if raw_file.find('.pkl') == -1:continue
      process_search_episode_file( raw_file,raw_dir,key_configs,processed_dir )
    import  pdb;pdb.set_trace()

    #aggregate_data( processed_dir, 'place' )
    #agg_data = pickle.load(open(processed_dir+'aggregated_data_place_n_'\
    #                       +str(agg_data[0])+'.p','r'))
    #print 'n_data is', np.array(agg_data[0]).shape
  else:
    # parallel case
    raw_file_idx = int(sys.argv[1])
    raw_file = raw_file_list[raw_file_idx]
    if not os.path.isfile(raw_dir+raw_file):
      print 'do not have the raw file for this idx'
      return
    process_search_episode_file( raw_file,raw_dir,key_configs,processed_dir )

if __name__ == '__main__':
  main()
  #test_get_training_data()
  #test_get_trajectories()

