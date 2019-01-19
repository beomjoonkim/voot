import numpy as np
import sys

from problem_environment import ProblemEnvironment
from problem_environments.mover_problem import MoverProblem

## openrave_wrapper imports
from manipulation.bodies.bodies import set_color
from manipulation.primitives.savers import DynamicEnvironmentStateSaver
from openravepy import *

## mover library utility functions
sys.path.append('../mover_library/')
from utils import *
from motion_planner import collision_fn

OBJECT_ORIGINAL_COLOR = (0, 0, 0)
COLLIDING_OBJ_COLOR = (0, 1, 1)
TARGET_OBJ_COLOR = (1, 0, 0)


class Mover(ProblemEnvironment):
    def __init__(self):
        ProblemEnvironment.__init__(self)

        problem = MoverProblem(self.env)
        self.problem_config = problem.get_problem_config()
        self.robot = self.env.GetRobots()[0]

        self.objects = self.problem_config['objects']
        for obj in self.objects:
            set_color(obj, OBJECT_ORIGINAL_COLOR)

        self.regions = {'entire_region': self.problem_config['entire_region']}

        self.init_base_conf = self.problem_config['init_base_config']
        self.goal_base_conf = self.problem_config['goal_base_config']
        self.infeasible_reward = 0 #-1
        self.name = 'mcr'
        self.init_saver = DynamicEnvironmentStateSaver(self.env)
        self.collision_fn = collision_fn(self.env, self.robot)

    def get_objs_in_region(self, region_name):
        movable_objs = self.objects
        objs_in_region = []
        for obj in movable_objs:
            if self.regions[region_name].contains(obj.ComputeAABB()):
                objs_in_region.append(obj)
        return objs_in_region

    def apply_next_base_pose(self, action, node, check_feasibility, parent_motion):
        if action['base_pose'] is None:
            return None, self.infeasible_reward, None, []
        xytheta_delta = action['action_parameters']
        new_robot_config = action['base_pose'] + xytheta_delta
        new_robot_config = clean_pose_data(new_robot_config).squeeze()
        if abs(new_robot_config[-1]) > 2*np.pi:
            import pdb;pdb.set_trace()
        prev_robot_config =get_body_xytheta(self.robot)
        set_robot_config(new_robot_config, self.robot)


        curr_state = None
        reward = self.determine_reward(prev_robot_config, new_robot_config)
        return curr_state, reward, new_robot_config, []

    def get_region_containing(self, obj):
        return self.regions['entire_region']

    def determine_reward(self, prev, new):
        # this reward function does not prevent slow progress
        prev_dist_to_goal = se2_distance(prev, self.goal_base_conf, 1, 1)
        new_dist_to_goal = se2_distance(new, self.goal_base_conf, 1, 1)
        distance_reward = -new_dist_to_goal # 1 if prev_dist_to_goal - new_dist_to_goal > 0 else 0
        collision_reward = -(self.collision_fn(new))*1
        return distance_reward + collision_reward

    def reset_to_init_state(self, node):
        node.state_saver.Restore()
        #set_robot_config(self.init_base_conf, self.robot)

    def disable_objects(self):
        for object in self.objects:
            object.Enable(False)

    def enable_objects(self):
        for object in self.objects:
            object.Enable(True)

    def check_base_pose_feasible(self, base_pose, obj, region):
        pass

    def is_goal_reached(self):
        return len(self.objs_to_move) == 0

    def which_operator(self, obj=None):
        return 'next_base_pose'

    def apply_action(self):
        pass







