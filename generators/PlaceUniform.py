import sys
import numpy as np
import pickle

sys.path.append('../mover_library/')
from samplers import *
from utils import *
from utils import place_distance
from planners.mcts_utils import make_action_executable


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
            self.robot.SetTransform(original_trans)
            self.robot.SetDOFValues(original_config)
            release_obj(self.robot, obj)
            with self.robot:
                #print target_obj_region
                obj_pose = randomly_place_in_region(self.env, obj, target_obj_region)  # randomly place obj

                # compute the resulting robot transform
                new_T_robot = np.dot(obj.GetTransform(), T_r_wrt_o)
                self.robot.SetTransform(new_T_robot)
                self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
                robot_xytheta = self.robot.GetActiveDOFValues()
                set_robot_config(robot_xytheta, self.robot)
                new_T = self.robot.GetTransform()
                assert (np.all(np.isclose(new_T, new_T_robot)))
                #if not (check_collision_except(obj, self.env)) \
                grab_obj(self.robot, obj)
                if not (self.env.CheckCollision(obj) or self.env.CheckCollision(self.robot)) \
                        and (target_robot_region.contains(self.robot.ComputeAABB())):
                    self.robot.SetTransform(original_trans)
                    self.robot.SetDOFValues(original_config)
                    return {'operator_name': 'two_arm_place', 'base_pose': robot_xytheta, 'object_pose': obj_pose}

        #import pdb;pdb.set_trace()
        return None

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
