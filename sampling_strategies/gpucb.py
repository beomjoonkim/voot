from sampling_strategy import SamplingStrategy
from planners.mcts_utils import make_action_executable


class GPUCB(SamplingStrategy):
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

        evaled_x = []
        evaled_y = []
        for x, y in zip(node.Q.keys(), node.Q.values()):
            if None in x:
                continue
            x = make_action_executable(x)
            if operator == 'two_arm_pick':
                evaled_x.append(x['pick_parameters'])
            else:
                evaled_x.append(x['object_pose'])
            evaled_y.append(y)
        if operator == 'two_arm_pick':
            evaled_x = [pick['pick_parameters'] for pick in node.all_evaled_q.keys()]
            action = self.pick_pi.predict(obj, region, evaled_x, evaled_y, 100)
        elif operator == 'two_arm_place':
            evaled_x = [pick['obj_pose'] for pick in node.all_evaled_q.keys()]
            action = self.place_pi.predict(obj, region, evaled_x, evaled_y, 100)
        else:
            assert False, "Undefined operator name"

        return action
