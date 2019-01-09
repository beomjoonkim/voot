import sys
import numpy as np

sys.path.append('../mover_library/')
from samplers import sample_pick, sample_grasp_parameters, sample_ir, sample_ir_multiple_regions

from utils import compute_occ_vec, set_robot_config, remove_drawn_configs, \
    draw_configs, clean_pose_data, draw_robot_at_conf, \
    check_collision_except, two_arm_pick_object, two_arm_place_object, pick_distance, place_distance


from planners.mcts_utils import make_action_executable
from PickGenerator import PickGenerator


class PickWithBaseUnif(PickGenerator):
    def __init__(self, problem_env):
        PickGenerator.__init__(self, problem_env)
        self.environment_name = problem_env.name

    def sample_closest_to_best_action(self, obj, region, best_action, other_actions):
        best_dist = np.inf
        other_dists = np.array([-1])
        counter = 1
        while np.any(best_dist > other_dists):

            if len(other_dists) > 0:
                print "Gaussian pick sampling, counter", counter
                best_action_base_pose = best_action['base_pose']
                var_base_pose = np.array([0.3, 0.3, 0.5]) / float(counter)
                pick_base_pose = np.random.normal(best_action_base_pose, var_base_pose)
                # todo
                #   skip the base pose on the samples that is in collision

                best_action_grasp_params = best_action['grasp_params']
                var_grasp = np.array([0.5, 0.2, 0.2]) / float(counter)
                grasp_params = np.random.normal(best_action_grasp_params, var_grasp)

                if grasp_params[0] > np.pi:
                    grasp_params[0] = np.pi
                elif grasp_params[0] < np.pi/4.0:
                    grasp_params[0] = np.pi/4.0

                if grasp_params[1] > 1:
                    grasp_params[1] = 1
                elif grasp_params[1] < 0:
                    grasp_params[1] = 0

                if grasp_params[2] > 1:
                    grasp_params[2] = 1
                elif grasp_params[2] < 0:
                    grasp_params[2] = 0
            else:
                with self.robot:
                    pick_base_pose = sample_ir(obj, self.robot, self.env, region)
                if pick_base_pose is None:
                    return None, None
                theta, height_portion, depth_portion = sample_grasp_parameters()
                grasp_params = np.array([theta[0], height_portion[0], depth_portion[0]])
            counter += 1

            pick_params = {'operator_name': 'two_arm_pick', 'base_pose': pick_base_pose, 'grasp_params': grasp_params}
            best_dist = pick_distance(pick_params, best_action, obj)
            other_dists = np.array([pick_distance(other, pick_params, obj) for other in other_actions])
        return pick_base_pose, grasp_params

    def predict_closest_to_best_action(self, obj, region, best_action, other_actions, n_iter):
        best_action = make_action_executable(best_action)
        other_actions = [make_action_executable(a) for a in other_actions]

        for iter in range(n_iter):
            print "Sampling closest"
            pick_base_pose, grasp_params = self.sample_closest_to_best_action(obj, region, best_action, other_actions)

            if pick_base_pose is None:
                continue

            if self.problem_env.name == 'convbelt':
                self.problem_env.disable_objects()
                obj.Enable(True)
            g_config = self.compute_g_config(obj, pick_base_pose, grasp_params)
            if self.problem_env.name == 'convbelt':
                self.problem_env.enable_objects()

            if g_config is not None:
                pick_action = {'operator_name': 'two_arm_pick', 'base_pose': pick_base_pose,
                               'grasp_params': grasp_params, 'g_config': g_config}
                return pick_action

        print "Sampling pick failed"
        pick_action = {'operator_name': 'two_arm_pick', 'base_pose': None, 'grasp_params': None, 'g_config': None}
        return pick_action

    def predict(self, obj, region, n_iter):
        if self.problem_env.is_solving_namo:
            pick_params = self.compute_grasp_action(obj, region, n_iter)
        else:
            self.problem_env.disable_objects()
            obj.Enable(True)
            pick_params = self.compute_grasp_action(obj, region, n_iter)
            self.problem_env.enable_objects()

        return pick_params





