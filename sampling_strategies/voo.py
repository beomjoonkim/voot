import numpy as np
import sys

sys.path.append('./mover_library/')
from sampling_strategy import SamplingStrategy
from utils import get_body_xytheta


class VOO(SamplingStrategy):
    def __init__(self, environment, pick_pi, place_pi, explr_p):
        SamplingStrategy.__init__(self, environment, pick_pi, place_pi)
        self.explr_p = explr_p

    def grasp_distance(self, a1, a2, curr_obj):
        obj_xyth = get_body_xytheta(curr_obj)
        grasp_a1 = np.array(a1[0])
        base_a1 = np.array(a1[1])
        relative_config_a1 = base_a1 - obj_xyth

        grasp_a2 = np.array(a2[0])
        base_a2 = np.array(a2[1])
        relative_config_a2 = base_a2 - obj_xyth
        return np.sum(abs(grasp_a1-grasp_a2)) + np.sum(self.base_conf_distance(relative_config_a1, relative_config_a2))

    @staticmethod
    def base_conf_distance(x, y):
        return np.sum(abs(x - y))

    def sample_from_best_voroi_region(self, evaled_actions, evaled_scores, which_operator):
        best_action = evaled_actions[np.argmax(evaled_scores)]

        if len(np.unique(evaled_scores)) == 1:
            if which_operator == 'two_arm_pick':
                action = self.pick_pi.predict(self.environment.curr_obj)
            else:
                action = self.place_pi.predict(self.environment.curr_obj)
            return action

        if which_operator == 'two_arm_pick':
            action = self.pick_pi.predict(self.environment.curr_obj)
            dists_to_non_best_actions = np.array([self.grasp_distance(action, y, self.environment.curr_obj)
                                                  for y in evaled_actions if y != best_action])
            dist_to_curr_best_action = np.array(self.grasp_distance(action, best_action, self.environment.curr_obj))
        else:
            action = self.place_pi.predict(self.environment.curr_obj)
            dists_to_non_best_actions = np.array([self.base_conf_distance(action, y)
                                                  for y in evaled_actions if y != best_action])
            dist_to_curr_best_action = np.array(self.base_conf_distance(action, best_action))

        n_trials = 0
        while len(dists_to_non_best_actions) != 0 and np.any(dist_to_curr_best_action > dists_to_non_best_actions) \
                and n_trials < 30:
            if which_operator == 'two_arm_pick':
                action = self.pick_pi.predict(self.environment.curr_obj)
                dist_to_curr_best_action = np.array(self.grasp_distance(action, best_action, self.environment.curr_obj))
            else:
                action = self.place_pi.predict(self.environment.curr_obj)
                dist_to_curr_best_action = np.array(self.base_conf_distance(action, best_action))
            n_trials += 1

            print "Is pick?", which_operator == 'two_arm_pick'
            print "Sampling from best voronoi region. Best and other action distances, and n_trial ", \
                dist_to_curr_best_action, dists_to_non_best_actions, n_trials

        return action

    def sample_from_uniform(self, which_operator):
        if which_operator == 'two_arm_pick':
            action = self.pick_pi.predict(self.environment.curr_obj)
        else:
            action = self.place_pi.predict(self.environment.curr_obj)
        return action

    def sample_next_point(self, node, which_operator):
        evaled_scores = node.Q.values()
        evaled_actions = node.Q.keys()

        rnd = np.random.random()
        if rnd < 1-self.explr_p and len(evaled_actions) > 0 \
                and np.max(evaled_scores) > self.environment.infeasible_reward:
            print "VOO sampling from best voronoi region"
            action = self.sample_from_best_voroi_region(evaled_actions, evaled_scores, which_operator)
        else:
            print 'n actions taken in this node, ', len(evaled_actions)
            if len(evaled_actions)>0:
                print 'best_score in this node,', np.max(evaled_scores)
            print "VOO sampling from uniform"
            action = self.sample_from_uniform(which_operator)

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

