from sampling_strategy import SamplingStrategy


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

        evaled_x = node.all_evaled_q.keys()
        evaled_y = node.all_evaled_q.values()

        if operator == 'two_arm_pick':
            action = self.pick_pi.predict(obj, region, evaled_x, evaled_y, 1000)
        elif operator == 'two_arm_place':
            action = self.place_pi.predict(obj, region, evaled_x, evaled_y, 1000)
        else:
            assert False, "Undefined operator name"

        return action
