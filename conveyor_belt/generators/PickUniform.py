import sys
import copy

sys.path.append('../mover_library/')
from samplers import sample_pick, sample_grasp_parameters, sample_ir

from utils import compute_occ_vec, set_robot_config, remove_drawn_configs, \
    draw_configs, clean_pose_data, draw_robot_at_conf, \
    pick_obj, place_obj, check_collision_except

sys.path.append('../mover_library/')
# from utils import *
from operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp


def check_collision_except_obj(obj, robot, env):
    in_collision = (check_collision_except(obj, robot, env)) \
                   or (check_collision_except(robot, obj, env))
    return in_collision


class PickUnif:
    def __init__(self, problem_env, robot, all_region):
        self.problem_env = problem_env
        self.openrave_env = problem_env.problem['env']
        self.robot = robot
        self.all_region = all_region

    def predict(self, obj):
        leftarm_manip = self.robot.GetManipulator('leftarm')
        rightarm_manip = self.robot.GetManipulator('rightarm')
        rightarm_torso_manip = self.robot.GetManipulator('rightarm_torso')

        for _ in range(1000):
            pick_base_pose = None
            while pick_base_pose is None:
                with self.robot:
                    pick_base_pose = sample_ir(obj, self.robot, self.openrave_env, self.all_region)
            theta, height_portion, depth_portion = sample_grasp_parameters()
            grasp_params = [theta[0], height_portion[0], depth_portion[0]]

            with self.robot:
                set_robot_config(pick_base_pose, self.robot)
                grasps = compute_two_arm_grasp(depth_portion=grasp_params[2],
                                               height_portion=grasp_params[1],
                                               theta=grasp_params[0],
                                               obj=obj,
                                               robot=self.robot)

                g_config = solveTwoArmIKs(self.openrave_env, self.robot, obj, grasps)
                if g_config is not None:
                    pick_obj(obj, self.robot, g_config, leftarm_manip, rightarm_torso_manip)
                    set_robot_config(self.problem_env.init_base_conf, self.robot)

                    if not check_collision_except(obj, self.robot, self.openrave_env):
                        set_robot_config(pick_base_pose, self.robot)
                        place_obj(obj, self.robot, leftarm_manip, rightarm_manip)
                        action = (pick_base_pose, grasp_params)
                        return action

                    set_robot_config(pick_base_pose, self.robot)
                    place_obj(obj, self.robot, leftarm_manip, rightarm_manip)

        return None
