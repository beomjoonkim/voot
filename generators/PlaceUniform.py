import sys
import numpy as np
import pickle

sys.path.append('../mover_library/')
from samplers import *
from utils import *
from utils import place_distance
from planners.mcts_utils import make_action_executable
import time

def generate_rand(min, max):
    return np.random.rand() * (max - min) + min


class PlaceUnif:
    def __init__(self, problem_env):
        self.problem_env = problem_env
        self.env = problem_env.env
        self.robot = self.env.GetRobots()[0]
        self.robot_region = self.problem_env.regions['entire_region']

    def predict(self, obj, obj_region):
        original_trans = self.robot.GetTransform()
        original_config = self.robot.GetDOFValues()
        T_r_wrt_o = np.dot(np.linalg.inv(obj.GetTransform()), self.robot.GetTransform())

        # obj_region is the task-level object region - where you want it to be in the task plan
        if self.problem_env.is_solving_namo:
            target_obj_region = self.problem_env.get_region_containing(obj)
            target_robot_region = target_obj_region # for namo, you want to stay in the same region
        else:
            target_robot_region = self.robot_region
            target_obj_region = obj_region # for fetching, you want to move it around

        print "Sampling place"
        for _ in range(1000):
            obj_pose, robot_xytheta = self.get_placement(obj, target_obj_region, T_r_wrt_o)
            set_robot_config(robot_xytheta, self.robot)
            if not (self.env.CheckCollision(obj) or self.env.CheckCollision(self.robot)) \
                    and (target_robot_region.contains(self.robot.ComputeAABB())):
                self.robot.SetTransform(original_trans)
                self.robot.SetDOFValues(original_config)
                return {'operator_name': 'two_arm_place', 'base_pose': robot_xytheta, 'object_pose': obj_pose}
            else:
                self.robot.SetTransform(original_trans)
                self.robot.SetDOFValues(original_config)

        self.robot.SetTransform(original_trans)
        self.robot.SetDOFValues(original_config)
        print "Sampling failed"
        return {'operator_name': 'two_arm_place', 'base_pose': None, 'object_pose': None}

    def get_placement(self, obj, target_obj_region, T_r_wrt_o):
        original_trans = self.robot.GetTransform()
        original_config = self.robot.GetDOFValues()
        self.robot.SetTransform(original_trans)
        self.robot.SetDOFValues(original_config)

        release_obj(self.robot, obj)
        with self.robot:
            # print target_obj_region
            obj_pose = randomly_place_in_region(self.env, obj, target_obj_region)  # randomly place obj
            obj_pose = obj_pose.squeeze()

            # compute the resulting robot transform
            new_T_robot = np.dot(obj.GetTransform(), T_r_wrt_o)
            self.robot.SetTransform(new_T_robot)
            self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
            robot_xytheta = self.robot.GetActiveDOFValues()
            set_robot_config(robot_xytheta, self.robot)
            grab_obj(self.robot, obj)
        return obj_pose, robot_xytheta

    def get_gaussian_placement_wrt_target_obj_placement(self, obj, target_obj_placement, target_obj_region, T_r_wrt_o, variance):
        original_trans = self.robot.GetTransform()
        original_config = self.robot.GetDOFValues()
        self.robot.SetTransform(original_trans)
        self.robot.SetDOFValues(original_config)

        release_obj(self.robot, obj)
        with self.robot:
            # print target_obj_region
            obj_pose = gaussian_randomly_place_in_region(self.env, obj, target_obj_region, center=target_obj_placement, var=variance)  # randomly place obj
            obj_pose = obj_pose.squeeze()

            # compute the resulting robot transform
            new_T_robot = np.dot(obj.GetTransform(), T_r_wrt_o)
            self.robot.SetTransform(new_T_robot)
            self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
            robot_xytheta = self.robot.GetActiveDOFValues()
            set_robot_config(robot_xytheta, self.robot)
            grab_obj(self.robot, obj)
        return obj_pose, robot_xytheta

    def sample_closest_to_best_action(self, obj, target_obj_region, best_action, other_actions, T_r_wrt_o):
        best_dist = np.inf
        other_dists = np.array([-1])
        counter = 1
        stime = time.time()
        while np.any(best_dist > other_dists):
            #if obj.GetName() != 'obj0' or counter > 10:
            #    print time.time()-stime
            #    import pdb;pdb.set_trace()

            if len(other_dists) > 0:
                variance = np.array([0.3, 0.3, 0.5]) / counter
                variance = 2*(np.array([0.3, 0.3, 0.5]) / counter)
                #print 'Gaussian sampling', best_dist, other_dists, variance
                obj_pose, robot_xytheta = self.get_gaussian_placement_wrt_target_obj_placement(obj,
                                                                                               best_action['object_pose'],
                                                                                               target_obj_region,
                                                                                               T_r_wrt_o,
                                                                                               variance)
            else:
                print 'Regular sampling', best_dist, other_dists
                obj_pose, robot_xytheta = self.get_placement(obj, target_obj_region, T_r_wrt_o)

            action = {'operator_name': 'two_arm_place', 'base_pose': robot_xytheta, 'object_pose': obj_pose}
            best_dist = place_distance(action, best_action, obj)
            other_dists = np.array([place_distance(other, action, obj) for other in other_actions])
            counter += 1

        return obj_pose, robot_xytheta

    def predict_closest_to_best_action(self, obj, obj_region, best_action, other_actions):
        best_action = make_action_executable(best_action)
        other_actions = [make_action_executable(a) for a in other_actions]

        original_trans = self.robot.GetTransform()
        original_config = self.robot.GetDOFValues()
        T_r_wrt_o = np.dot(np.linalg.inv(obj.GetTransform()), self.robot.GetTransform())

        # obj_region is the task-level object region - where you want it to be in the task plan
        if self.problem_env.is_solving_namo:
            target_obj_region = self.problem_env.get_region_containing(obj)
            target_robot_region = target_obj_region  # for namo, you want to stay in the same region
        else:
            target_robot_region = self.robot_region
            target_obj_region = obj_region  # for fetching, you want to move it around

        for iter in range(1000):
            print "Sampling place iter: ", iter
            obj_pose, robot_xytheta = self.sample_closest_to_best_action(obj, obj_region, best_action, other_actions, T_r_wrt_o)
            print "Done sampling closest"

            set_robot_config(robot_xytheta, self.robot)
            if not (self.env.CheckCollision(obj) or self.env.CheckCollision(self.robot)) \
                    and (target_robot_region.contains(self.robot.ComputeAABB())):
                self.robot.SetTransform(original_trans)
                self.robot.SetDOFValues(original_config)
                action = {'operator_name': 'two_arm_place', 'base_pose': robot_xytheta, 'object_pose': obj_pose}
                print "Found best placement"
                return action
            else:
                self.robot.SetTransform(original_trans)
                self.robot.SetDOFValues(original_config)

        self.robot.SetTransform(original_trans)
        self.robot.SetDOFValues(original_config)
        print "Sampling failed"
        return {'operator_name': 'two_arm_place', 'base_pose': None, 'object_pose': None}

    """
    def predict_closest_to_best_action(self, obj, obj_region, best_action, other_actions):
        best_action = make_action_executable(best_action)
        other_actions = [make_action_executable(a) for a in other_actions]

        best_dist = np.inf
        other_dists = np.array([-1])
        while np.any(best_dist > other_dists):
            action = self.predict(obj, obj_region)
            best_dist = place_distance(action, best_action, obj)
            other_dists = np.array([place_distance(other, action, obj) for other in other_actions])

        return action
    """
