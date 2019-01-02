# this script generates the key configurations in this environment
# the key configurations are used as an input to each action

# if we are going to condition on the key config, then why do we need
# time step feature?

# the answer depends on how long it takes to check collisions at the key configs
from NAMO_problem import NAMO_problem
from manipulation.primitives.transforms import get_point,set_point
from manipulation.bodies.bodies import set_config
from manipulation.bodies.bodies import box_body, randomly_place_body, place_xyz_body
from manipulation.motion.primitives import extend_fn,distance_fn,sample_fn,collision_fn


from openravepy import *
import sys
import os
import pickle
import numpy as np
sys.path.append('../mover_library/')
from utils import set_robot_config,draw_configs
from key_config_utils import *
from sklearn import preprocessing

def load_data(ftrain):
  data = pickle.load( open('./train_data/'+ftrain,'r') )
  return data

def get_nodes(data):
  nodes = data['nodes']
  pick_nodes = [n for n in nodes if n.sample != None and 'g_config' in n.sample.keys()]
  place_nodes = [n for n in nodes if n.sample != None and 'place_base_pose' in n.sample.keys()]
  return pick_nodes,place_nodes

def get_fetch_path( data  ):
  raw_path = data['original_path']
  stepsize = int(round(0.1 * len(raw_path),0))
  idxs = range(1,len(raw_path),stepsize)
  path = [raw_path[idx] for idx in idxs]
  if not np.all(path[-1] == raw_path[-1]): 
    path.append(raw_path[-1])

  return [path]

def join_all_paths( pick,place,fetch ):
  return pick+place+fetch

def get_all_configs():
  n_data = 0
  configs = []
  for ftrain in os.listdir('./train_data'):
    is_non_data_file = os.path.isdir('./train_data/'+ftrain)
    if is_non_data_file: continue

    data = load_data( ftrain )
    is_bad_data = (len(data['nodes']) < 50) and \
                 (np.max(np.array(data['rwd_time_list'])[:,1])\
                 <np.max(np.array(data['rwd_time_list'])[:,2]))
    if is_bad_data: 
      continue

    pick_nodes,place_nodes = get_nodes(data)

    pick_paths  = get_paths(pick_nodes) 
    place_paths = get_paths(place_nodes)
    fetch_path  = get_fetch_path( data )
    all_paths   = join_all_paths( pick_paths,place_paths,fetch_path )

    get_configs_from_paths( all_paths,configs )

    # stop adding data
    n_data +=1
    print n_data, len(configs)
    if n_data % 10 == 0:
      pickle.dump( configs,\
                  open('key_configs.p','wb'))
  import pdb;pdb.set_trace()

def visualize_konfs():
  env = Environment()
  env.SetViewer('qtcoin')
  problem = NAMO_problem(env,compute_orig_path=False)
  konfs = np.array(pickle.load(open('key_configs.p','r')))
  draw_configs(konfs,env,name='pick',colors=(1,0,0))
  import pdb;pdb.set_trace()
  """
  pick_confs = np.array(pickle.load(open('pick_key_configs.p','r')))
  draw_configs(pick_confs,env,name='pick',colors=(1,0,0))
  place_confs = np.array(pickle.load(open('place_key_configs.p','r')))
  draw_configs(place_confs,env,name='place',colors=(0,0,1))
  """
def test_get_configs_from_paths():
  place_confs = np.array(pickle.load(open('place_evaluator/place_key_configs.p','r')))
  place_confs = np.array(pickle.load(open('./place_key_configs.p','r')))
  """
  flist = os.listdir('./train_data')
  for ftrain in flist:    
    # test if all paths are looked at
    print ftrain
    data   = pickle.load( open('./train_data/'+ftrain,'r') )
    nodes  = data['nodes']
    pick_nodes = [n for n in nodes if n.sample != None and 'g_config' in n.sample.keys()]
    pick_paths = get_paths(pick_nodes) # remember I plan path from where I placed
    
    pick_configs= []
    get_configs_from_paths( pick_paths, pick_configs)

    for p in pick_configs:
      same_conf = [q for path in pick_paths for q in path if np.all(p==q)]
      assert(len(same_conf)==1)
    print 'Test0: All configs considered passed'

    xy_threshold = 0.2
    th_threshold = 20*np.pi/180
    for p in pick_configs:
      for q in pick_configs:
        if np.all(p==q): continue
        try: 
          assert( (np.linalg.norm(p[0:2]-q[0:2]) >= xy_threshold) \
                 or (abs(p[2]-q[2]) >= th_threshold) )
        except AssertionError:      
          print 'dist failed'
          import pdb;pdb.set_trace() 
  """
  for p in place_confs:
    try:
      assert(p[-1] >= 0 and p[-1] <= 2*np.pi)
    except AssertionError:
      print 'angle threshold failed'
      import pdb;pdb.set_trace()
  import pdb;pdb.set_trace()
  print "PASSED"
        
  
def main():
  #test_get_configs_from_paths()  
  #visualize_konfs()
  get_all_configs() 
  print 'finihsed generating configs'

if __name__ == '__main__':
  main()


