# this script generates the key configurations in this environment
# the key configurations are used as an input to each action

# if we are going to condition on the key config, then why do we need
# time step feature?

# the answer depends on how long it takes to check collisions at the key configs
from conveyor_belt_problem import two_tables_through_door
from manipulation.primitives.transforms import get_point,set_point,quat_from_z_rot,set_quat
from manipulation.bodies.bodies import set_config
from manipulation.bodies.bodies import box_body, randomly_place_body, place_xyz_body
from manipulation.motion.primitives import extend_fn,distance_fn,sample_fn,collision_fn


from openravepy import *


import sys
import os
import pickle
import numpy as np
sys.path.append('../mover_library/')
from key_config_utils import *
from samplers import *
from sklearn import preprocessing
from sklearn.cluster import KMeans


def get_all_configs():
  n_data = 0
  train_data_dir = './train_data/'
  configs = []
  train_fs =  os.listdir(train_data_dir)
  np.random.shuffle(train_fs)
  for ftrain in train_fs:
    if ftrain.find('.pkl') == -1: continue
    # load the data
    data = pickle.load( open(train_data_dir+ftrain,'r') )
    if 'nodes' not in data.keys(): continue

    # get all the paths
    nodes = data['nodes']
    paths = get_paths(nodes) # remember I plan path from where I placed

    # get all configurations in all paths
    get_configs_from_paths( paths,configs )
    #draw_configs(configs,problem['env'],name='c',colors=(0,0,1),transparency=0)

    # stop adding data
    n_data +=1
    if len(configs)>10000:
      break
    print n_data, len(configs)
  configs=np.array(configs)
  kmeans = KMeans(n_clusters=int(0.1*len(configs)), random_state=170).fit(configs)
  configs = configs[np.argsort(kmeans.predict(configs))] 
  pickle.dump(configs,open('./place_soap/key_configs.p','wb'))
  return configs

def visualize_konfs():
  env=Environment()
  env.SetViewer('qtcoin')
  problem=two_tables_through_door(env,obj_shapes={},obj_poses={},obst_shapes={},obst_poses={})
  env.Remove(env.GetRobots()[0])
  key_configs = pickle.load(open('./place_soap/key_configs.p','r'))
  draw_configs(key_configs,env)

def main():
  get_all_configs()
  visualize_konfs()

if __name__ == '__main__':
  main()

