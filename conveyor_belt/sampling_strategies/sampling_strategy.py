class SamplingStrategy:
    def __init__(self, environment, pick_pi, place_pi):
        self.environment = environment
        self.pick_pi = pick_pi
        self.place_pi = place_pi

    def sample_next_point(self, node):
        raise NotImplementedError
