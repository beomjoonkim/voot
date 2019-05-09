from trajectory_representation.operator import Operator

import pickle
import numpy as np

from deap import benchmarks


class SyntheticEnv:
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
        self.problem_idx = problem_idx
        if problem_idx == 0:
            self.dim_x = 3
            self.feasible_action_value_threshold = 1.0
        elif problem_idx == 1:
            self.dim_x = 10
            self.feasible_action_value_threshold = 0.1
        elif problem_idx == 2:
            self.dim_x = 20
            self.feasible_action_value_threshold = 0.5

        config = pickle.load(
            open('./test_results/function_optimization/shekel/shekel_dim_' + str(self.dim_x) + '.pkl', 'r'))
        A = config['A']
        C = config['C']
        self.reward_function = lambda sol: benchmarks.shekel(sol, A, C)[0]

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
        if reward < self.feasible_action_value_threshold:
            reward = reward + self.infeasible_reward
            # todo stop advancing if your reward is less than 0.3
            operator_instance.continuous_parameters['is_feasible'] = False
        else:
            reward += 1.0
            operator_instance.continuous_parameters['is_feasible'] = True

        return reward

    def is_action_feasible(self, action):
        reward = self.apply_action_and_get_reward(action, True, None)
        return reward > self.feasible_action_value_threshold

    def is_goal_reached(self):
        return False

    def get_applicable_op_skeleton(self, parent_action):
        op = Operator(operator_type='synthetic_'+str(self.dim_x),
                      discrete_parameters={},
                      continuous_parameters=None,
                      low_level_motion=None)
        return op

    def is_pick_time(self):
        return False




