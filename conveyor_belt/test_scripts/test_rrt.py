import sys

from openravepy import *
from conveyor_belt_env import ConveyorBelt
from generators.PickUniform import PickUnif

sys.path.append('../mover_library/')
from utils import draw_robot_at_conf, remove_drawn_configs, visualize_path
from motion_planner import collision_fn, extend_fn,  distance_fn, sample_fn, rrt_connect, base_extend_fn, \
    base_sample_fn, smooth_path, base_distance_fn

import numpy as np
import time
from openravepy import *

def main():
    convbelt = ConveyorBelt(v=True)
    problem = convbelt.problem
    pick_pi = PickUnif(convbelt, problem['env'].GetRobots()[0], problem['all_region'])
    pick_action = pick_pi.predict(convbelt.curr_obj)
    convbelt.apply_pick_action(pick_action)

    convbelt.env.SetViewer('qtcoin')
    env = convbelt.env
    robot = convbelt.robot
    robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    base_loc = np.array([-2,-2, np.pi/2.])
    base_loc = np.array([-1.1,-1, np.pi/2])
    #draw_robot_at_conf(base_loc, 0.5, 'cg', robot, env, color=None)

    d_fn = base_distance_fn(robot, x_extents=2.51, y_extents=2.51)
    s_fn = base_sample_fn(robot, x_extents=2.51, y_extents=2.51)
    e_fn = base_extend_fn(robot)
    c_fn = collision_fn(env, robot)

    q1 = robot.GetActiveDOFValues()
    q2 = base_loc

    path = None
    import pdb;pdb.set_trace()
    while path is None:
        path = rrt_connect(q1, q2, d_fn, s_fn, e_fn, c_fn, iterations=20) # 10 rand config
    path = smooth_path(path, e_fn, c_fn)
    visualize_path(robot, path)
    print path
    import pdb;pdb.set_trace()
    remove_drawn_configs('cg', env)


if __name__ == '__main__':
    main()