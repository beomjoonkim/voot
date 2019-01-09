from PickGenerator import PickGenerator
from gpucb_utils.gp import StandardContinuousGP
from gpucb_utils.functions import UCB, Domain
from gpucb_utils.bo import BO

import sys
sys.path.append('../mover_library/')
from utils import *
from samplers import compute_robot_xy_given_ir_parameters
import numpy as np


class PickGPUCB(PickGenerator):
    def __init__(self, problem_env):
        PickGenerator.__init__(self, problem_env)
        self.robot = problem_env.robot
        self.env = problem_env.env
        self.problem_env = problem_env

        self.gp = StandardContinuousGP(6)
        self.acq_fcn = UCB(zeta=0.01, gp=self.gp)
        pick_domain = self.get_pick_domain()
        self.pick_optimizer = BO(self.gp, self.acq_fcn, pick_domain)  # this depends on the object

    @staticmethod
    def get_pick_domain():
        portion_domain = [[0.4], [0.9]]
        base_angle_domain = [[0], [2 * np.pi]]
        facing_angle_domain = [[-30 * np.pi / 180.0], [30 * np.pi / 180]]
        base_pose_domain = np.hstack([portion_domain, base_angle_domain, facing_angle_domain])

        grasp_param_domain = np.array([[45 * np.pi / 180, 0.5, 0.1], [180 * np.pi / 180, 1, 0.9]])
        domain = Domain(0, np.hstack([grasp_param_domain, base_pose_domain]))
        return domain

    @staticmethod
    def get_grasp_and_base_pose_params(obj, pick_parameters):
        grasp_params = pick_parameters[0:3]
        portion = pick_parameters[3]
        base_angle = pick_parameters[4]
        facing_angle = pick_parameters[5]
        pick_base_pose = compute_robot_xy_given_ir_parameters(portion, base_angle, obj)

        obj_xy = get_point(obj)[:-1]
        robot_xy = pick_base_pose[0:2]
        angle_to_be_set = compute_angle_to_be_set(obj_xy, robot_xy)
        pick_base_pose[-1] = angle_to_be_set + facing_angle
        return grasp_params, pick_base_pose

    def predict(self, obj, region, evaled_x, evaled_y, n_iter):
        for i in range(n_iter):
            pick_parameters = self.pick_optimizer.choose_next_point(evaled_x, evaled_y)
            grasp_params, pick_base_pose = self.get_grasp_and_base_pose_params(obj, pick_parameters)
            set_robot_config(pick_base_pose, self.robot)
            g_config = self.compute_g_config(obj, pick_base_pose, grasp_params)  # this checks collision
            if g_config is not None:
                pick_action = {'operator_name': 'two_arm_pick', 'base_pose': pick_base_pose,
                               'grasp_params': grasp_params, 'g_config': g_config, 'pick_parameters':pick_parameters}
                return pick_action
            else:
                evaled_x.append(pick_parameters)
                evaled_y.append(self.problem_env.infeasible_reward)

        pick_action = {'operator_name': 'two_arm_pick', 'base_pose': None,
                       'grasp_params': None, 'g_config': None}
        return pick_action

