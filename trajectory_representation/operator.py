# This class describes an operator, in terms of:
#   type, discrete parameters (represented with entity class instance), continuous parameteres,
#   and the associated low-level motions


class Operator:
    def __init__(self, operator_type, discrete_parameters, continuous_parameters, low_level_motion):
        self.type = operator_type

        assert type(discrete_parameters) is dict, "Discrete parameters of an operator must be a dictionary"
        self.discrete_parameters = discrete_parameters
        self.continuous_parameters = continuous_parameters
        self.low_level_motion = None

    def update_low_level_motion(self, low_level_motion):
        self.low_level_motion = low_level_motion


