import sys
import numpy as np

sys.path.append('../mover_library/')
from samplers import sample_pick, sample_grasp_parameters, sample_ir, sample_ir_multiple_regions

from utils import compute_occ_vec, set_robot_config, remove_drawn_configs, \
    draw_configs, clean_pose_data, draw_robot_at_conf, \
    check_collision_except, two_arm_pick_object, two_arm_place_object, pick_distance, place_distance

sys.path.append('../mover_library/')
from operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp
from planners.mcts_utils import make_action_executable



#def check_collision_except_obj(obj, robot, env):
#    in_collision = (check_collision_except(obj, robot, env)) \
#                   or (check_collision_except(robot, obj, env))
#    return in_collision


class PickUnif(object):
    def __init__(self, problem_env):
        self.problem_env = problem_env
        self.env = problem_env.env
        self.robot = self.env.GetRobots()[0]

    def predict(self, obj, region):
        raise NotImplementedError


class PickWithBaseUnif(PickUnif):
    def __init__(self, problem_env):
        PickUnif.__init__(self, problem_env)
        self.environment_name = problem_env.name

    def compute_grasp_config(self, obj, pick_base_pose, grasp_params):
        set_robot_config(pick_base_pose, self.robot)
        grasps = compute_two_arm_grasp(depth_portion=grasp_params[2],
                                       height_portion=grasp_params[1],
                                       theta=grasp_params[0],
                                       obj=obj,
                                       robot=self.robot)

        g_config = solveTwoArmIKs(self.env, self.robot, obj, grasps)
        return g_config

    def compute_grasp_action(self, obj, region, n_iter=100):
        leftarm_manip = self.robot.GetManipulator('leftarm')
        rightarm_manip = self.robot.GetManipulator('rightarm')
        rightarm_torso_manip = self.robot.GetManipulator('rightarm_torso')

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
        pick_action = {'operator_name': 'two_arm_pick', 'base_pose': pick_base_pose, 'grasp_params': grasp_params, 'g_config': g_config}
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
                    if not check_collision_except(obj, self.env):
                        two_arm_place_object(obj, self.robot, pick_action)
                        set_robot_config(self.problem_env.init_base_conf, self.robot)
                        print "Sampling pick succeeded"
                        return g_config
                    else:
                        two_arm_place_object(obj, self.robot, pick_action)
                        set_robot_config(self.problem_env.init_base_conf, self.robot)
                else:
                    two_arm_pick_object(obj, self.robot, pick_action)
                    if not check_collision_except(obj, self.env):
                        two_arm_place_object(obj, self.robot, pick_action)
                        print "Sampling pick succeeded"
                        return g_config
                    else:
                        two_arm_place_object(obj, self.robot, pick_action)
            else:
                return None

    def sample_closest_to_best_action(self, obj, region, best_action, other_actions):
        best_dist = np.inf
        other_dists = np.array([-1])
        while np.any(best_dist > other_dists):
            with self.robot:
                pick_base_pose = sample_ir(obj, self.robot, self.env, region)
            if pick_base_pose is None:
                return None, None
            theta, height_portion, depth_portion = sample_grasp_parameters()
            grasp_params = np.array([theta[0], height_portion[0], depth_portion[0]])
            pick_params = {'operator_name': 'two_arm_pick', 'base_pose': pick_base_pose, 'grasp_params': grasp_params}
            best_dist = pick_distance(pick_params, best_action, obj)
            other_dists = np.array([pick_distance(other, pick_params, obj) for other in other_actions])
        return pick_base_pose, grasp_params


    def predict_closest_to_best_action(self, obj, region, best_action, other_actions):
        #pick_params = self.compute_grasp_action_closest_to_best_action(obj, region, n_iter=1000)
        best_action = make_action_executable(best_action)
        other_actions = [make_action_executable(a) for a in other_actions]

        for iter in range(1000):
            pick_base_pose, grasp_params = self.sample_closest_to_best_action(obj, region, best_action, other_actions)
            if pick_base_pose is None:
                continue
            if self.problem_env.name == 'convbelt':
                self.problem_env.disable_objects()
            g_config = self.compute_g_config(obj, pick_base_pose, grasp_params)
            if self.problem_env.name == 'convbelt':
                self.problem_env.enable_objects()

            if g_config is not None:
                pick_action = {'operator_name': 'two_arm_pick', 'base_pose': pick_base_pose,
                               'grasp_params': grasp_params, 'g_config': g_config}
                return pick_action

        print "Sampling pick failed"
        pick_action = {'operator_name': 'two_arm_pick', 'base_pose': pick_base_pose, 'grasp_params': grasp_params, 'g_config': g_config}
        return pick_action

    def predict(self, obj, region):
        if self.problem_env.is_solving_namo:
            if self.problem_env.name == 'namo':
                pick_params = self.compute_grasp_action(obj, region, n_iter=100)
            else:
                pick_params = self.compute_grasp_action(obj, region, n_iter=100)
        else:
            if self.problem_env.name == 'convbelt':
                self.problem_env.disable_objects()
                pick_params = self.compute_grasp_action(obj, region, n_iter=1000)
                self.problem_env.enable_objects()
            else:
                #pick_params = self.compute_grasp_action(obj, region, n_iter=10)
                pick_params = self.compute_grasp_action(obj, region, n_iter=10)

        if self.problem_env.is_solving_fetching and pick_params['g_config'] is None:
            self.problem_env.disable_objects_in_region(region.name)
            obj.Enable(True)
            pick_params = self.compute_grasp_action(obj, region, n_iter=100)
            self.problem_env.enable_objects_in_region(region.name)


        return pick_params





