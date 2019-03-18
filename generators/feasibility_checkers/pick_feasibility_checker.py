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
        # This function checks if the base pose is not in collision and if there is a feasible pick
        obj = node.operator_skeleton.discrete_parameters['object']
        grasp_params, pick_base_pose = get_pick_base_pose_and_grasp_from_pick_parameters(obj, pick_parameters)
        g_config = self.compute_g_config(obj, pick_base_pose, grasp_params)

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
        # todo this part, I need to ignore
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
                two_arm_pick_object(obj, self.robot, pick_action)

                inside_region = self.problem_env.regions['entire_region'].contains(self.robot.ComputeAABB())

                if not self.env.CheckCollision(self.robot) and inside_region:
                    two_arm_place_object(obj, self.robot, pick_action)
                    return g_config
                else:
                    two_arm_place_object(obj, self.robot, pick_action)
            else:
                return None

