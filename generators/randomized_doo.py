import sys
import numpy as np

sys.path.append('../mover_library/')
from generator import Generator
from planners.mcts_utils import make_action_executable

from utils import pick_parameter_distance, place_parameter_distance
from doo_utils.doo_tree import BinaryDOOTree
from utils import pick_parameter_distance, place_parameter_distance
from doo import DOOGenerator

import matplotlib.pyplot as plt
import copy


class RandomizedDOOGenerator(DOOGenerator):
    def __init__(self, node, problem_env, explr_p):
        DOOGenerator.__init__(self, node, problem_env, explr_p)
        self.dim_x = self.domain[0].shape[-1]

    def choose_next_point(self):
        next_node = self.doo_tree.get_next_point_and_node_to_evaluate()
        x_to_evaluate = np.random.uniform(next_node.cell_min, next_node.cell_max, (1, self.dim_x)).squeeze()
        next_node.evaluated_x = x_to_evaluate
        x_to_evaluate = self.unnormalize_x_value(x_to_evaluate)
        return x_to_evaluate, next_node

