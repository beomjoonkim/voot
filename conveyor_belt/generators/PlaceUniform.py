import sys
import numpy as np
import pickle

sys.path.append('../mover_library/')
from samplers import *
from utils import *


def generate_rand(min, max):
    return np.random.rand() * (max - min) + min


class PlaceUnif():
    def __init__(self, env, robot, obj_region, robot_region):
        self.key_configs = pickle.load(open('./key_configs/key_configs.p', 'r'))
        self.env = env
        self.robot = robot
        self.obj_region = obj_region
        self.robot_region = robot_region

    '''
    def predict(self, state, n_samples=1):
        import pdb;pdb.set_trace()
        x = generate_rand(-2.51, 1)
        y = generate_rand(-2.51, 2.51)
        th = generate_rand(-np.pi, np.pi)
        return np.array([[x, y, th]])
    '''

    def predict(self, obj, n_samples=1):
        original_trans = self.robot.GetTransform()
        T_r_wrt_o = np.dot(np.linalg.inv(obj.GetTransform()), self.robot.GetTransform())
        samples = []

        for _ in range(1000):
            release_obj(self.robot, obj)
            self.robot.SetTransform(original_trans)

            # first sample collision-free object placement
            with self.robot:
                obj_xytheta = randomly_place_in_region(self.env, obj, self.obj_region)  # randomly place obj

                # compute the resulting robot transform
                new_T_robot = np.dot(obj.GetTransform(), T_r_wrt_o)
                self.robot.SetTransform(new_T_robot)
                self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
                robot_xytheta = self.robot.GetActiveDOFValues()
                set_robot_config(robot_xytheta, self.robot)
                new_T = self.robot.GetTransform()
                assert (np.all(np.isclose(new_T, new_T_robot)))
                if not (check_collision_except(self.robot, obj, self.env)) \
                        and (self.robot_region.contains(self.robot.ComputeAABB())):
                    grab_obj(self.robot, obj)
                    samples.append(robot_xytheta)
                    if len(samples) == n_samples:
                        return np.array(samples)
        return None