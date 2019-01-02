import sys

sys.path.append('../mover_library/')
from utils import *
from samplers import sample_grasp_parameters, sample_ir, randomly_place_in_region

from NAMO_env import NAMO
from data_load_utils import get_sars_data
from openravepy import *
from manipulation.bodies.bodies import randomly_place_region, place_body, place_body_on_floor
from manipulation.primitives.transforms import get_point, set_point, pose_from_quat_point, unit_quat


class UniformSampler(object):
    def __init__(self, env, obj_region, robot_region):
        self.obj_region = obj_region
        self.robot_region = robot_region
        self.env = env
        self.robot = self.env.GetRobots()[0]

    def predict(self, fc, misc, n_samples=1):
        raise NotImplemented


class UniformPick(UniformSampler):
    def __init__(self, env, obj_region, robot_region):
        super(UniformPick, self).__init__(env, obj_region, robot_region)

    def predict(self, obj, fc, misc, n_samples=1):
        actions = []
        with self.robot:
            for _ in range(n_samples):
                base_pose = sample_ir(obj, self.robot, self.env, self.robot_region)
                theta, height_portion, depth_portion = sample_grasp_parameters()
                # while base_pose is None:
                #  base_pose                        = sample_ir(obj,self.robot,self.env,self.robot_region)
                action = np.hstack([theta, height_portion, depth_portion, base_pose])[None, :]
                actions.append(action)
        actions = np.array(actions).reshape(n_samples, 6)
        return actions


class UniformPlace(UniformSampler):
    def __init__(self, env, obj_region, robot_region):
        super(UniformPlace, self).__init__(env, obj_region, robot_region)

    def predict(self, fc, misc, n_samples=1):
        # Predict robot base pose
        actions = []
        min_x = self.robot_region.box[0][0]
        max_x = self.robot_region.box[0][1]
        min_y = self.robot_region.box[1][0]
        max_y = self.robot_region.box[1][1]

        for _ in range(n_samples):
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            th = np.random.uniform(0, 2 * pi)
            action = np.array([x, y, th])
            # with self.robot:
            #  action=randomly_place_in_region(self.env,self.robot,self.robot_region)
            actions.append(action)
        actions = np.array(actions).reshape(n_samples, 3)
        return actions


def rollout():
    traj_list = []
    for n_iter in range(5):
        problem = NAMO()  # different "initial" state
        place_pi = UniformPlace(problem.problem['env'], \
                                problem.problem['obj_region'], \
                                problem.problem['all_region'])
        pick_pi = UniformPick(problem.problem['env'], \
                              problem.problem['obj_region'], \
                              problem.problem['all_region'])
        print "Executing policy..."
        traj = problem.execute_policy(pick_pi, place_pi, 10, key_configs=[], visualize=False)
        traj_list.append(traj)
        problem.env.Destroy()
        RaveDestroy()
        print traj['r']
    return traj_list


def test_uniform_policies():
    testing = False
    avg_Js = []
    for n in range(300):
        traj_list = rollout()
        avg_J = np.mean([np.sum(traj['r']) for traj in traj_list])
        std_J = np.std([np.sum(traj['r']) for traj in traj_list])
        print "Avg J,std J, and n", avg_J, std_J, n
        avg_Js.append(avg_J)
        print "Avg Js", avg_Js
    print "Avg Js", avg_Js


if __name__ == '__main__':
    results = [1.4, 1.0, 3.2, 1.2, 3.6, 3.6, 3.2, 0.2, 1.0, -0.2, -1.4, -0.2, -0.6, 1.4, 0.2, 2.2, 4.2, 1.8, 1.8, -0.6,
               0.6, \
               -0.4, -0.2, 0.6, 1.0, 0.0, -0.8, 0.2, 3.0, -0.2, -0.8, 0.4, -0.6, -0.2, -1.6, 1.8, -0.2, 2.4, 1.0, 1.4,
               1.6, 0.2, 0.6, 0.4, \
               0.8, -0.2, 0.2, 0.0, 1.8, 1.8, -2.0, -2.0, 0.6, 0.6, 1.0, 1.4, 0.2, 1.0, 0.6, 1.0, 2.2, 1.4, 0.0, 1.0,
               -1.6]

    test_uniform_policies()
    print "Max, mean, std", np.max(asdf), np.mean(asdf), np.std(asdf)
    import pdb;

    pdb.set_trace()
    test_uniform_policies()
