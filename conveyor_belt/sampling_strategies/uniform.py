from sampling_strategy import SamplingStrategy

class Uniform(SamplingStrategy):
    def __init__(self, environment, pick_pi, place_pi):
        SamplingStrategy.__init__(self, environment, pick_pi, place_pi)

    def sample_next_point(self, node):
        is_pick_node = node.parent is None or len(node.parent_action) != 2
        if is_pick_node:
            action = self.pick_pi.predict(self.environment.curr_obj)
        else:
            action = self.place_pi.predict(self.environment.curr_obj)

        return action
