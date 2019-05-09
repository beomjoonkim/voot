from trajectory_representation.operator import Operator
from problem_environment import ProblemEnvironment

import pickle
import numpy as np

from deap import benchmarks



class SyntheticEnv():
    def __init__(self, problem_idx):
        # I want to create a sequence of Shekel functions as a reward function
        # Each state is associated with a shekel function
        # Then, I need a way to:
        #   1) Store state and reward pairs
        #   2) Select the next state based on the current action
        #   3) Restore the state-reward pairs
        self.name = 'synthetic'
        self.env = None
        self.robot = None
        self.objects_currently_not_in_goal = []
        self.infeasible_reward = -2
        if problem_idx == 0:
            dim_x = 3
            config = pickle.load(
                open('./test_results/function_optimization/shekel/shekel_dim_' + str(dim_x) + '.pkl', 'r'))
            A = config['A']
            C = config['C']
            self.reward_function = lambda sol: benchmarks.shekel(sol, A, C)[0]

    def is_pick_time(self):
        return False

    def reset_to_init_state(self, node):
        # todo reset to the original state. Do this by changing the reward function to the initial one.
        pass

    def apply_action_and_get_reward(self, operator_instance, is_op_feasible, node):
        action = operator_instance.continuous_parameters['action_parameters']
        # todo make the action to change the next state's reward function
        #       how should I change it?
        #       one simple idea is to shift the shekel function around
        #
        reward = self.reward_function(action)
        return reward

    def apply_operator_instance(self, operator_instance, node):
        reward = self.apply_action_and_get_reward(operator_instance, True, node)
        if reward < 0.3:
            reward = reward + self.infeasible_reward
            # todo stop advancing if your reward is less than 0.3
        return reward

    def is_goal_reached(self):
        return False

    def get_applicable_op_skeleton(self, parent_action):
        op = Operator(operator_type='synthetic_action',
                      discrete_parameters={},
                      continuous_parameters=None,
                      low_level_motion=None)
        return op



