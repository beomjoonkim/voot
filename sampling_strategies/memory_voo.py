import numpy as np
import sys

sys.path.append('./mover_library/')
from sampling_strategy import SamplingStrategy
from utils import get_body_xytheta, clean_pose_data, pick_distance, place_distance
from planners.mcts_utils import make_action_executable


# todo I don't think I need this file
class MemoryVOO(SamplingStrategy):
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
        evaled_scores = node.evaled_x
        evaled_actions = node.evaled_y


        rnd = np.random.random()
        is_sample_from_best_v_region = rnd < 1-self.explr_p and len(evaled_actions) > 1 \
                                           and np.max(evaled_scores) > self.environment.infeasible_reward

        if is_sample_from_best_v_region:
            print "VOO sampling from best voronoi region"
            action = self.sample_from_best_voroi_region(evaled_actions, evaled_scores, node)
        else:
            print "VOO sampling from uniform"
            action = self.sample_from_uniform(node)

        return action



