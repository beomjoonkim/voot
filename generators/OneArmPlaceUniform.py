import sys
import numpy as np
import pickle

sys.path.append('../mover_library/')
from samplers import *
from utils import *
from operator_utils.grasp_utils import solveIK, compute_one_arm_grasp


def generate_rand(min, max):
    return np.random.rand() * (max - min) + min


class OneArmPlaceUnif:
    def __init__(self, problem_env):
        self.env = problem_env.env
        self.problem_env = problem_env
        self.robot = self.env.GetRobots()[0]

    def place_robot_on_floor(self):
        FLOOR_Z = 0.136183
        trans = self.robot.GetTransform()
        trans[2, -1] = FLOOR_Z
        self.robot.SetTransform(trans)

    def predict(self, grasp_params, obj, obj_region):
        target_obj_region = obj_region

        # use the same grasp parameters?

        T_r_wrt_o = np.dot(np.linalg.inv(obj.GetTransform()), self.robot.GetTransform())
        for _ in range(1000):
            release_obj(self.robot, obj)

            with self.robot:
                print "Sampling obj placement"
                obj_xytheta = randomly_place_in_region(self.env, obj, target_obj_region)  # randomly place obj
                print "Done"
                new_T_robot = np.dot(obj.GetTransform(), T_r_wrt_o)
                self.robot.SetTransform(new_T_robot)
                self.place_robot_on_floor()
                place_base_pose = get_body_xytheta(self.robot).squeeze()
                if not self.problem_env.regions['entire_region'].contains(self.robot.ComputeAABB()):
                    continue

                # find an IK solution with grasp parameters; because I am maintaining my relative base pose, it should
                # be the same one as before
                grasps = compute_one_arm_grasp(depth_portion=grasp_params[2],
                                               height_portion=grasp_params[1],
                                               theta=grasp_params[0],
                                               obj=obj,
                                               robot=self.robot)

                for g in grasps:
                    g_config = self.robot.GetManipulator('rightarm_torso').FindIKSolution(g, 0)
                    if g_config is not None:
                        set_config(self.robot, g_config, self.robot.GetManipulator('rightarm_torso').GetArmIndices())
                        grab_obj(self.robot, obj) # put the robot state back to what it was before
                        if not self.env.CheckCollision(self.robot): #check_collision_except(obj, self.env):
                            return {'operator_name': 'one_arm_place', 'base_pose': place_base_pose, 'g_config': g_config}

        import pdb;pdb.set_trace()
        return {'operator_name': 'one_arm_place', 'base_pose': None, 'g_config': None}


