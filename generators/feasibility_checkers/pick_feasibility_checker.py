import sys
import numpy as np

sys.path.append('../mover_library/')

from utils import compute_occ_vec, set_robot_config, remove_drawn_configs, \
    draw_configs, clean_pose_data, draw_robot_at_conf, \
    check_collision_except, two_arm_pick_object, two_arm_place_object, pick_distance, place_distance, \
    compute_angle_to_be_set, get_body_xytheta, get_pick_base_pose_and_grasp_from_pick_parameters
from operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp


class PickFeasibilityChecker(object):
    def __init__(self, problem_env):
        self.problem_env = problem_env
        self.env = problem_env.env
        self.robot = self.env.GetRobots()[0]

    def check_feasibility(self, node, pick_parameters):
        obj = node.obj
        grasp_params, pick_base_pose = get_pick_base_pose_and_grasp_from_pick_parameters(obj, pick_parameters)
        # todo disable objects for fetch
        if self.problem_env.name == 'convbelt' or self.problem_env.is_solving_fetching:
            self.problem_env.disable_objects()
            obj.Enable(True)
        g_config = self.compute_g_config(obj, pick_base_pose, grasp_params)
        if self.problem_env.name == 'convbelt' or self.problem_env.is_solving_fetching:
            self.problem_env.enable_objects()

        if g_config is not None:
            pick_action = {'operator_name': 'two_arm_pick', 'base_pose': pick_base_pose,
                           'grasp_params': grasp_params, 'g_config': g_config, 'action_parameters': pick_parameters}
            return pick_action, 'HasSolution'
        else:
            pick_action = {'operator_name': 'two_arm_pick', 'base_pose': None, 'grasp_params': None, 'g_config': None,
                           'action_parameters': pick_parameters}
            return pick_action, "NoSolution"

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
                        #print "Sampling pick succeeded"
                        return g_config
                    else:
                        two_arm_place_object(obj, self.robot, pick_action)
                        set_robot_config(self.problem_env.init_base_conf, self.robot)
                else:
                    two_arm_pick_object(obj, self.robot, pick_action)
                    if not self.env.CheckCollision(self.robot):
                        two_arm_place_object(obj, self.robot, pick_action)
                        #print "Sampling pick succeeded"
                        return g_config
                    else:
                        two_arm_place_object(obj, self.robot, pick_action)
            else:
                return None
