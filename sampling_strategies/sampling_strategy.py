class SamplingStrategy:
    def __init__(self, environment, pick_pi, place_pi, one_arm_pick_pi=None, one_arm_place_pi=None):
        self.environment = environment
        self.pick_pi = pick_pi
        self.place_pi = place_pi
        self.one_arm_pick_pi = one_arm_pick_pi
        self.one_arm_place_pi = one_arm_place_pi

    def sample_next_point(self, node):
        raise NotImplementedError
