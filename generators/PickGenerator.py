import sys
import numpy as np

sys.path.append('../mover_library/')

from utils import compute_occ_vec, set_robot_config, remove_drawn_configs, \
    draw_configs, clean_pose_data, draw_robot_at_conf, \
    check_collision_except, two_arm_pick_object, two_arm_place_object, pick_distance, place_distance
from samplers import sample_pick, sample_grasp_parameters, sample_ir, sample_ir_multiple_regions
from operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp


class PickGenerator(object):
    def __init__(self, problem_env):
        self.problem_env = problem_env
        self.env = problem_env.env
        self.robot = self.env.GetRobots()[0]

    def predict(self, obj, region, n_iter):
        raise NotImplementedError

    def compute_grasp_config(self, obj, pick_base_pose, grasp_params):
        set_robot_config(pick_base_pose, self.robot)
        if self.env.CheckCollision(self.robot):
            return None
        grasps = compute_two_arm_grasp(depth_portion=grasp_params[2],
                                       height_portion=grasp_params[1],
                                       theta=grasp_params[0],
                                       obj=obj,
                                       robot=self.robot)

        g_config = solveTwoArmIKs(self.env, self.robot, obj, grasps)
        return g_config

    def compute_grasp_action(self, obj, region, n_iter):
        grasp_params = None
        g_config = None

        print 'Sampling pick'
        for iter in range(n_iter):
            # sample pick parameters
            with self.robot:
                pick_base_pose = sample_ir(obj, self.robot, self.env, region)
            if pick_base_pose is None:
                return {'operator_name': 'two_arm_pick', 'base_pose': None, 'grasp_params': None, 'g_config': None}
            theta, height_portion, depth_portion = sample_grasp_parameters()
            grasp_params = np.array([theta[0], height_portion[0], depth_portion[0]])

            # compute if parameters are feasible
            g_config = self.compute_g_config(obj, pick_base_pose, grasp_params)
            if g_config is not None:
                pick_action = {'operator_name': 'two_arm_pick', 'base_pose': pick_base_pose,
                               'grasp_params': grasp_params, 'g_config': g_config}
                return pick_action
        print "Sampling pick failed"
        pick_action = {'operator_name': 'two_arm_pick', 'base_pose': None, 'grasp_params': None, 'g_config': None}
        return pick_action

    def compute_g_config(self, obj, pick_base_pose, grasp_params):
        with self.robot:
            g_config = self.compute_grasp_config(obj, pick_base_pose, grasp_params)
            if g_config is not None:
                pick_action = {'operator_name': 'two_arm_pick', 'base_pose': pick_base_pose,
                               'grasp_params': grasp_params, 'g_config': g_config}
                if self.problem_env.name == 'convbelt':
                    two_arm_pick_object(obj, self.robot, pick_action)
                    set_robot_config(self.problem_env.init_base_conf, self.robot)
                    if not self.env.CheckCollision(self.robot):
                        two_arm_place_object(obj, self.robot, pick_action)
                        set_robot_config(self.problem_env.init_base_conf, self.robot)
                        print "Sampling pick succeeded"
                        return g_config
                    else:
                        two_arm_place_object(obj, self.robot, pick_action)
                        set_robot_config(self.problem_env.init_base_conf, self.robot)
                else:
                    two_arm_pick_object(obj, self.robot, pick_action)
                    if not self.env.CheckCollision(self.robot):
                        two_arm_place_object(obj, self.robot, pick_action)
                        print "Sampling pick succeeded"
                        return g_config
                    else:
                        two_arm_place_object(obj, self.robot, pick_action)
            else:
                return None
