import os
import sys
sys.path.append('../mover_library/')
from preprocessing_utils import *
from samplers import *
from conveyor_belt_problem import two_tables_through_door


train_dir = './processed_train_data_with_pick_soap/'
aggregate_data(train_dir,'place')
env = Environment()
env.SetViewer('qtosg')
problem = two_tables_through_door(env,obst_shapes={},obst_poses={},obj_shapes={},obj_poses={})

n_data=1000
data =pickle.load(open( train_dir+'/place_aggregated_data.p')) 
x_data = np.array(data[0])[:n_data,:3] # actions - predict object pose
s_data = np.array(data[1])[:n_data,:] # scores
colors = np.zeros((len(s_data),1))
colors = np.hstack((np.exp(s_data),colors,colors))
colors = colors/np.max(colors)
print x_data.shape
draw_configs(x_data,env,colors=colors)

import pdb;pdb.set_trace()


