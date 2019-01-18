from problem_environments.mover_env import MoverProblem
import sys
sys.path.append('./mover_library/')

from utils import *
import numpy as np

problem = MoverProblem()
robot = problem.robot
robot_xytheta = get_body_xytheta(robot).squeeze()
np.random([-0.2,-0.2,-20*np.pi/180.], [-0.2,-0.2,-20*np.pi/180.],)
min_xytheta = [-0.2, -0.2, -20*np.pi/180.]
max_xytheta = [0.2, 0.2, 20*np.pi/180.]

x_to_evaluate = robot_xytheta + np.random.uniform(min_xytheta, max_xytheta, (1, 3)).squeeze()



import pdb;pdb.set_trace()