import sys

sys.path.append('../mover_library/')
from mover_library.motion_planner import collision_fn
from utils import get_body_xytheta, set_robot_config


class BasePoseFeasibilityChecker(object):
    def __init__(self, problem_env):
        self.problem_env = problem_env
        self.env = problem_env.env
        self.robot = self.env.GetRobots()[0]
        self.collision_fn = collision_fn(self.env, self.robot)

    def check_feasibility(self, node, action):
        robot_xytheta = get_body_xytheta(self.robot).squeeze()
        new_q = robot_xytheta + action

        self.problem_env.disable_objects() # note that this class is only for mcr purpose
        set_robot_config(new_q, self.problem_env.robot)
        if self.collision_fn(new_q) or \
                (not self.problem_env.regions['entire_region'].contains(self.robot.ComputeAABB())):
            action = {'operator_name': 'next_base_pose', 'base_pose': None,
                      'action_parameters': action}
            status = "NoSolution"
        else:
            action = {'operator_name': 'next_base_pose', 'base_pose': robot_xytheta,
                      'action_parameters': action}
            status = 'HasSolution'
        self.problem_env.enable_objects()
        set_robot_config(robot_xytheta, self.problem_env.robot)

        return action, status


