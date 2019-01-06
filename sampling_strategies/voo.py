import numpy as np
import sys

sys.path.append('./mover_library/')
from sampling_strategy import SamplingStrategy
from utils import get_body_xytheta, clean_pose_data, pick_distance, place_distance
from planners.mcts_utils import make_action_executable


class VOO(SamplingStrategy):
    def __init__(self, environment, pick_pi, place_pi, explr_p):
        SamplingStrategy.__init__(self, environment, pick_pi, place_pi)
        self.explr_p = explr_p

    def sample_from_best_voroi_region(self, evaled_actions, evaled_scores, node):
        best_action = evaled_actions[np.argmax(evaled_scores)]
        which_operator = node.operator
        curr_obj = node.obj
        region = node.region

        if which_operator == 'two_arm_pick':
            action = self.pick_pi.predict_closest_to_best_action(curr_obj, region, best_action, evaled_actions)
        else:
            action = self.place_pi.predict_closest_to_best_action(curr_obj, region, best_action, evaled_actions)

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

