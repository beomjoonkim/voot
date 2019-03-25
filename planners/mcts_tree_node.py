import numpy as np
from mcts_utils import is_action_hashable, make_action_hashable, make_action_executable
from trajectory_representation.operator import Operator


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
        self.objs_in_collision = None
        self.n_ucb_iterations = 0

        self.sampling_strategy = sampling_strategy

    def get_never_evaluated_action(self):
        # get list of actions that do not have an associated Q values
        no_evaled = [a for a in self.A if a not in self.Q.keys()]
        return np.random.choice(no_evaled)

    def is_ucb_step(self, widening_parameter, infeasible_rwd):
        n_arms = len(self.A)
        if n_arms < 5:
            return False
        else:
            all_explored_actions_are_infeasible = np.max(self.reward_history.values()) == infeasible_rwd
            # should there be more than one action? I do not think so
            if all_explored_actions_are_infeasible:
                return False

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
            for action, value in zip(self.Q.keys(), self.Q.values()):
                ucb_value = value + self.ucb_parameter * upper_confidence_bound(self.Nvisited, self.N[action])

                # todo randomized tie-break
                if ucb_value > best_value:
                    best_action = action
                    best_value = ucb_value

        return best_action

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



