from sampling_strategy import SamplingStrategy
from planners.mcts_utils import make_action_executable


class DOO:
    def __init__(self, environment, pick_pi, place_pi):
        SamplingStrategy.__init__(self, environment, pick_pi, place_pi)
        self.robot = environment.robot
        self.env = environment.env
        self.problem_env = environment

    def sample_next_point(self, node):
        # fit GP-UCB
        obj = node.obj
        region = node.region
        operator = node.operator

        if operator == 'two_arm_pick':
            action = self.pick_pi.predict(obj, region, node, 100)
        elif operator == 'two_arm_place':
            action = self.place_pi.predict(obj, region, node, 100)
        else:
            assert False, "Undefined operator name"

        return action
