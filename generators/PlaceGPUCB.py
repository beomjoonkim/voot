from PickGenerator import PickGenerator
from gpucb_utils.gp import StandardContinuousGP
from gpucb_utils.functions import UCB, Domain
from gpucb_utils.bo import BO

import sys
sys.path.append('../mover_library/')
from utils import *
import numpy as np


class PlaceGPUCB:
    def __init__(self, problem_env):
        self.problem_env = problem_env
        self.env = problem_env.env
        self.robot = self.env.GetRobots()[0]
        self.robot_region = self.problem_env.regions['entire_region']
        self.place_acq_fcn = UCB(zeta=0.01, gp=self.place_gp)
        self.place_gp = StandardContinuousGP(3)

        if problem_env.name == 'convbelt':
            place_domain = self.get_place_domain(problem_env.regions['object_region'])
        else:
            place_domain = self.get_place_domain(problem_env.regions['entire_region'])

        self.place_optimizer = BO(self.place_gp, self.place_acq_fcn, place_domain)  # this depends on the problem

    def predict(self,  obj, obj_region, evaled_x, evaled_y, n_iter):
        T_r_wrt_o = np.dot(np.linalg.inv(obj.GetTransform()), self.robot.GetTransform())
        if self.problem_env.is_solving_namo:
            target_obj_region = self.problem_env.get_region_containing(obj)
            target_robot_region = target_obj_region  # for namo, you want to stay in the same region
        else:
            target_robot_region = self.problem_env.regions['entire_region']
            target_obj_region = obj_region  # for fetching, you want to move it around

        for i in range(n_iter):
            obj_pose = self.place_optimizer.choose_next_point(evaled_x, evaled_y)
            robot_xytheta = self.compute_robot_base_pose_given_object_pose(obj, self.robot, obj_pose, T_r_wrt_o)
            with self.robot:
                set_robot_config(robot_xytheta, self.robot)
                is_base_pose_feasible = not (
                            self.env.CheckCollision(obj) or self.env.CheckCollision(self.robot)) and \
                                        (target_robot_region.contains(self.robot.ComputeAABB())) and \
                                        (target_obj_region.contains(obj.ComputeAABB()))

            if is_base_pose_feasible:
                action = {'operator_name': 'two_arm_place', 'base_pose': robot_xytheta, 'object_pose': obj_pose}
                return action
            else:
                print evaled_x
            print len(evaled_x), len(evaled_y)
        return {'operator_name': 'two_arm_place', 'base_pose': None, 'object_pose': None}, evaled_x, evaled_y

    @staticmethod
    def get_place_domain(region):
        box = np.array(region.box)
        x_range = np.array([[box[0, 0]], [box[0, 1]]])
        y_range = np.array([[box[1, 0]], [box[1, 1]]])
        th_range = np.array([[0], [2 * np.pi]])
        domain = Domain(0, np.hstack([x_range, y_range, th_range]))
        return domain

    @staticmethod
    def compute_robot_base_pose_given_object_pose(obj, robot, obj_pose, T_r_wrt_o):
        original_obj_T = obj.GetTransform()
        original_robot_T = robot.GetTransform()

        release_obj(robot, obj)
        set_obj_xytheta(obj_pose, obj)
        new_T_robot = np.dot(obj.GetTransform(), T_r_wrt_o)
        robot.SetTransform(new_T_robot)
        robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
        robot_xytheta = robot.GetActiveDOFValues()
        grab_obj(robot, obj)
        robot.SetTransform(original_robot_T)
        return robot_xytheta