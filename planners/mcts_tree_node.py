import numpy as np
import sys
from mcts_utils import is_action_hashable, make_action_hashable, make_action_executable


def UCT_upperbound(n, n_sa):
    return 2 * np.log(n) / float(n_sa)



class TreeNode:
    def __init__(self, obj, region, operator, exploration_parameter, depth, state_saver, sampling_strategy, is_init_node):
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
        self.exploration_parameter = exploration_parameter
        self.parent_motion = None
        self.is_init_node = False
        self.is_goal_node = False
        self.is_goal_and_already_visited = False
        self.depth = depth
        self.sum_rewards = 0
        self.operator = None

        self.state_saver = state_saver
        self.obj = obj
        self.operator = operator
        self.region = region
        self.is_init_node = is_init_node
        self.objs_in_collision = None
        self.sampling_strategy = sampling_strategy

    def get_best_action(self):
        best_value = -np.inf
        for action, value in zip(self.Q.keys(), self.Q.values()):
            uct_value = value + self.exploration_parameter * UCT_upperbound(self.Nvisited, self.N[action])
            #print 'uct value:', value, self.exploration_parameter * UCT_upperbound(self.Nvisited, self.N[action])

            if uct_value > best_value:
                best_action = action
                best_value = uct_value

        executable_action = make_action_executable(best_action)
        """
        is_pick_action = len(best_action) == 2
        if is_pick_action:
            best_action = tuple((np.array(best_action[0]), np.array(best_action[1])))
        else:
            best_action = np.array(list(best_action)).squeeze()
        """
        return executable_action

    def is_action_tried(self, action):
        if action is None:
            return False
        if is_action_hashable(action):
            return action in self.A
        else:
            return make_action_hashable(action) in self.A

    def get_child_node(self, action):
        if is_action_hashable(action):
            return self.children[action]
        else:
            return self.children[make_action_hashable(action)]




