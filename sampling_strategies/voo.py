import numpy as np
import sys

sys.path.append('./mover_library/')
from sampling_strategy import SamplingStrategy
from utils import get_body_xytheta, clean_pose_data
from planners.mcts_utils import make_action_executable


class VOO(SamplingStrategy):
    def __init__(self, environment, pick_pi, place_pi, explr_p):
        SamplingStrategy.__init__(self, environment, pick_pi, place_pi)
        self.explr_p = explr_p

    def pick_distance(self, a1, a2, curr_obj):
        a2_executable = make_action_executable(a2)
        obj_xyth = get_body_xytheta(curr_obj)

        grasp_a1 = np.array(a1['grasp_params']).squeeze()
        base_a1 = clean_pose_data(np.array(a1['base_pose'])).squeeze()
        relative_config_a1 = (base_a1 - obj_xyth).squeeze()

        grasp_a2 = np.array(a2_executable['grasp_params']).squeeze()
        base_a2 = clean_pose_data(np.array(a2_executable['base_pose'])).squeeze()
        relative_config_a2 = (base_a2 - obj_xyth).squeeze()

        # normalize grasp distance
        grasp_max_diff = [1/2.356, 1., 1.]
        grasp_distance = np.sum( np.dot(abs(grasp_a1 - grasp_a2), grasp_max_diff))

        bas_distance_max_diff = np.array([1./(2*2.51), 1./(2*2.51), 1/2*np.pi])
        base_distance = np.sum(np.dot(self.base_conf_distance(relative_config_a1, relative_config_a2),
                                      bas_distance_max_diff))
        return grasp_distance + base_distance

    @staticmethod
    def base_conf_distance(x, y):
        return np.sum(abs(x - y))

    def place_distance(self, a1, a2, curr_obj):
        import pdb;pdb.set_trace()
        obj_xyth = get_body_xytheta(curr_obj)
        base_a1 = np.array(a1['base_pose'])
        relative_config_a1 = base_a1 - obj_xyth

        a2_executable = make_action_executable(a2)
        base_a2 = np.array(a2_executable['base_pose'])
        relative_config_a2 = base_a2 - obj_xyth
        bas_distance_max_diff = np.array([1. / (0.98), 1. / (0.98), 1 / 2 * np.pi])
        base_distance = np.sum(np.dot(self.base_conf_distance(relative_config_a1, relative_config_a2),
                                      bas_distance_max_diff))

        return base_distance

    def sample_from_best_voroi_region(self, evaled_actions, evaled_scores, node):
        best_action = evaled_actions[np.argmax(evaled_scores)]
        which_operator = node.operator
        curr_obj = node.obj
        region = node.region

        if which_operator == 'two_arm_pick':
            action = self.pick_pi.predict(curr_obj, region)
            dists_to_non_best_actions = np.array([self.pick_distance(action, y, curr_obj)
                                                  for y in evaled_actions if y != best_action])
            dist_to_curr_best_action = self.pick_distance(action, best_action, curr_obj)
        else:
            action = self.place_pi.predict(curr_obj, region)
            dists_to_non_best_actions = np.array([self.place_distance(action, y, curr_obj)
                                                  for y in evaled_actions if y != best_action])
            dist_to_curr_best_action = self.place_distance(action, best_action, curr_obj)

        print dist_to_curr_best_action, dists_to_non_best_actions
        n_trials = 0
        while len(dists_to_non_best_actions) != 0 and np.any(dist_to_curr_best_action > dists_to_non_best_actions) \
                and n_trials < 30:
            if which_operator == 'two_arm_pick':
                action = self.pick_pi.predict(curr_obj, region)
                dist_to_curr_best_action = np.array([self.pick_distance(action, best_action, curr_obj)])
            else:
                action = self.place_pi.predict(curr_obj, region)
                dist_to_curr_best_action = self.place_distance(action, best_action, curr_obj)
            n_trials += 1

            print "Is pick?", which_operator == 'two_arm_pick'

            print "Sampling from best voronoi region. Best and other action distances, and n_trial ", \
                dist_to_curr_best_action, dists_to_non_best_actions, n_trials
        #if len(dists_to_non_best_actions) > 1:
        #    import pdb;pdb.set_trace()

        return action

    def sample_from_uniform(self, node):
        obj = node.obj
        region = node.region
        which_operator = node.operator

        if which_operator == 'two_arm_pick':
            action = self.pick_pi.predict(obj, region)
        else:
            action = self.place_pi.predict(obj, region)
        return action

    def sample_next_point(self, node):
        evaled_scores = node.Q.values()
        evaled_actions = node.Q.keys()

        rnd = np.random.random()
        if rnd < 1-self.explr_p and len(evaled_actions) > 1 \
                and np.max(evaled_scores) > self.environment.infeasible_reward:
            print "VOO sampling from best voronoi region"
            action = self.sample_from_best_voroi_region(evaled_actions, evaled_scores, node)
        else:
            print 'n actions taken in this node, ', len(evaled_actions)
            if len(evaled_actions)>0:
                print 'best_score in this node,', np.max(evaled_scores)
            print "VOO sampling from uniform"
            action = self.sample_from_uniform(node)

        return action


class MoverVOO(VOO):
    def __init__(self, environment, two_arm_pick_pi, two_arm_place_pi, explr_p):
        VOO.__init__(self, environment, two_arm_pick_pi, two_arm_place_pi, explr_p)
        self.two_arm_pick_pi = two_arm_pick_pi
        self.two_arm_place_pi = two_arm_place_pi

    def sample_from_uniform(self, obj, region, which_operator):
        if which_operator == 'one_arm_pick':
            raise NotImplementedError
        elif which_operator == 'one_arm_place':
            raise NotImplementedError
        elif which_operator == 'two_arm_pick':
            action = self.two_arm_pick_pi.predict(obj,
                                                  region)

        elif which_operator == 'two_arm_place':
            action = self.two_arm_place_pi.predict(obj,
                                                   region)
        else:
            print "Invalid opreator name"
            sys.exit(-1)

        return action

    def sample_from_best_voroi_region(self, evaled_actions, evaled_scores, which_operator):
        pass

