import sys
import numpy as np
sys.path.append('./mover_library/')
from samplers import *
from utils import get_pick_domain, get_place_domain

from feasibility_checkers.pick_feasibility_checker import PickFeasibilityChecker
from feasibility_checkers.place_feasibility_checker import PlaceFeasibilityChecker
from feasibility_checkers.base_pose_feasibility_checker import BasePoseFeasibilityChecker
from planners.mcts_utils import make_action_executable


class Generator:
    def __init__(self, operator_name, problem_env):
        self.problem_env = problem_env
        self.env = problem_env.env
        self.evaled_actions = []
        self.evaled_q_values = []

        if operator_name == 'two_arm_pick':
            self.domain = get_pick_domain()
            self.feasibility_checker = PickFeasibilityChecker(problem_env)
        elif operator_name == 'two_arm_place':
            if problem_env.name == 'convbelt':
                place_domain = get_place_domain(problem_env.regions['object_region'])
            else:
                place_domain = get_place_domain(problem_env.regions['entire_region'])
                place_domain[0] = place_domain[0]*500
                place_domain[1][0:2] = place_domain[1][0:2]*500

            self.domain = place_domain
            self.feasibility_checker = PlaceFeasibilityChecker(problem_env)
        elif operator_name == 'next_base_pose':
            self.domain = np.array([[-0.2,-0.2,-20.*np.pi/180.0],[0.2,0.2,20*np.pi/180.0]])
            self.feasibility_checker = BasePoseFeasibilityChecker(self.problem_env)
        else:
            raise ValueError

    def update_evaled_values(self, node):
        executed_actions_in_node = node.Q.keys()
        executed_action_values_in_node = node.Q.values()

        for action, q_value in zip(executed_actions_in_node, executed_action_values_in_node):
            action_parameters = action.continuous_parameters['action_parameters']
            is_in_array = [np.array_equal(action_parameters, a) for a in self.evaled_actions]
            is_action_included = np.any(is_in_array)

            if not is_action_included:
                self.evaled_actions.append(action_parameters)
                self.evaled_q_values.append(q_value)
            else:
                # update the value if the action is included
                self.evaled_q_values[np.where(is_in_array)[0][0]] = q_value

    def sample_next_point(self, node, n_iter):
        raise NotImplementedError

    def sample_from_uniform(self):
        dim_parameters = self.domain.shape[-1]
        domain_min = self.domain[0]
        domain_max = self.domain[1]
        return np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()

