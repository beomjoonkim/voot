import numpy as np
import os
import pickle

from mcts_utils import is_action_hashable, make_action_hashable, make_action_executable
from trajectory_representation.operator import Operator
from mover_library.utils import visualize_path

import openravepy
def upper_confidence_bound(n, n_sa):
    return 2 * np.sqrt(np.log(n) / float(n_sa))


class TreeNode:
    def __init__(self, operator_skeleton, ucb_parameter, depth, state_saver, sampling_strategy,
                 is_init_node):
        self.Nvisited = 0
        self.N = {}  # N(n,a)
        self.Q = {}  # Q(n,a)
        self.A = []  # traversed actions
        self.parent = None
        self.children = {}
        self.parent_action = None
        self.sum_ancestor_action_rewards = 0  # for logging purpose
        self.sum_rewards_history = {}  # for debugging purpose
        self.reward_history = {}  # for debugging purpose
        self.ucb_parameter = ucb_parameter
        self.parent_motion = None
        self.is_goal_node = False
        self.is_goal_and_already_visited = False
        self.depth = depth
        self.sum_rewards = 0
        self.sampling_agent = None

        self.state_saver = state_saver
        self.operator_skeleton = operator_skeleton
        self.is_init_node = is_init_node
        self.objects_not_in_goal = None
        self.n_ucb_iterations = 0

        self.sampling_strategy = sampling_strategy
        self.is_goal_node = False
        self.idx = 1

        # for debugging purpose
        self.best_v = 0

    def get_never_evaluated_action(self):
        # get list of actions that do not have an associated Q values
        no_evaled = [a for a in self.A if a not in self.Q.keys()]
        return np.random.choice(no_evaled)

    def is_reevaluation_step(self, widening_parameter, infeasible_rwd, use_progressive_widening, use_ucb):
        n_arms = len(self.A)
        if n_arms < 1:
            return False

        max_reward_of_each_action = np.array([np.max(rlist) for rlist in self.reward_history.values()])
        n_feasible_actions = np.sum(max_reward_of_each_action > infeasible_rwd)
        next_state_terminal = np.any([c.is_goal_node for c in self.children.values()])

        if n_feasible_actions < 1 or next_state_terminal: # sample more actions
            return False

        if not use_ucb:
            new_action = self.A[-1]
            if np.max(self.reward_history[new_action]) <= -2:
                return False

        if use_progressive_widening:
            n_actions = len(self.A)
            is_time_to_sample = n_actions <= widening_parameter * self.Nvisited
            return is_time_to_sample
        else:
            if self.n_ucb_iterations < widening_parameter:
                self.n_ucb_iterations += 1
                return True
            else:
                self.n_ucb_iterations = 0
                return False

    def perform_ucb_over_actions(self):
        best_value = -np.inf
        never_executed_actions_exist = len(self.Q) != len(self.A)

        if never_executed_actions_exist:
            best_action = self.get_never_evaluated_action()
        else:
            best_action = self.Q.keys()[0]

            feasible_actions = [a for a in self.A if np.max(self.reward_history[a]) > -2]
            feasible_q_values = [self.Q[a] for a in feasible_actions]
            assert(len(feasible_actions) > 1)
            for action, value in zip(feasible_actions, feasible_q_values):
                ucb_value = value + self.ucb_parameter * upper_confidence_bound(self.Nvisited, self.N[action])

                # todo randomized tie-break
                if ucb_value > best_value:
                    best_action = action
                    best_value = ucb_value

        return best_action

    def make_actions_pklable(self):
        for a in self.A:
            if a.type == 'two_arm_pick' and type(a.discrete_parameters['object']) != str:
                a.discrete_parameters['object'] = str(a.discrete_parameters['object'].GetName())

    def make_actions_executable(self):
        assert len(openravepy.RaveGetEnvironments()) == 1
        env = openravepy.RaveGetEnvironment(1)
        for a in self.A:
            if a.type == 'two_arm_pick' and type(a.discrete_parameters['object']) == str:
                a.discrete_parameters['object'] = env.GetKinBody(a.discrete_parameters['object'])

    def store_node_information(self, domain_name):
        # state_saver, q_values, actions, reward_history, parent_action
        self.make_actions_pklable()
        fdir = './test_results/' + domain_name + '_results/visualization_purpose/'
        if not os.path.isdir(fdir):
            os.makedirs(fdir)

        to_store ={
            'Q': self.Q,
            'saver': self.state_saver,
            'progress': len(self.objects_not_in_goal)
        }
        pickle.dump(to_store, open(fdir + 'node_idx_' + str(self.idx) + '_' + self.sampling_strategy+'.pkl', 'wb'))
        self.make_actions_executable()

    def choose_new_arm(self):
        new_arm = self.A[-1]  # what to do if the new action is not a feasible one?
        is_new_arm_feasible = np.max(self.reward_history[new_arm]) > -2
        #is_new_arm_feasible = new_arm.continuous_parameters['base_pose'] is not None
        try:
            assert is_new_arm_feasible
        except:
            import pdb;pdb.set_trace()
        return new_arm

    def is_action_tried(self, action):
        return action in self.Q.keys()

    def get_child_node(self, action):
        if is_action_hashable(action):
            return self.children[action]
        else:
            return self.children[make_action_hashable(action)]

    def add_actions(self, continuous_parameters):
        new_action = Operator(operator_type=self.operator_skeleton.type,
                              discrete_parameters=self.operator_skeleton.discrete_parameters,
                              continuous_parameters=continuous_parameters,
                              low_level_motion=None)
        self.A.append(new_action)
        self.N[new_action] = 0


