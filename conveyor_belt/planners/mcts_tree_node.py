import numpy as np
from mcts_utils import is_action_hashable, make_action_hashable


def UCT_upperbound(n, n_sa):
    return 2 * np.log(n) / float(n_sa)


class TreeNode:
    def __init__(self, exploration_parameter, depth):
        self.Nvisited = 0
        self.N = {}  # N(n,a)
        self.Q = {}  # Q(n,a)
        self.A = []  # traversed actions
        self.parent = None
        self.children = {}
        self.parent_action = None
        self.sum_ancestor_action_rewards = 0  # for logging purpose
        self.sum_rewards_history = {}  # for debugging purpose
        self.exploration_parameter = exploration_parameter
        self.parent_grasp_or_path = None
        self.is_init_node = False
        self.is_goal_node = False
        self.depth = 0
        self.sum_rewards = 0

    def get_best_action(self):
        best_value = -np.inf
        for action, value in zip(self.Q.keys(), self.Q.values()):
            uct_value = value + self.exploration_parameter * UCT_upperbound(self.Nvisited, self.N[action])
            print 'uct value:', value, self.exploration_parameter * UCT_upperbound(self.Nvisited, self.N[action])
            if uct_value > best_value:
                best_action = action
                best_value = uct_value

        is_pick_action = len(best_action) == 2
        if is_pick_action:
            best_action = tuple((np.array(best_action[0]), np.array(best_action[1])))
        else:
            best_action = np.array([list(best_action)])
        return best_action

    def is_action_tried(self, action):
        if is_action_hashable(action):
            return action in self.A
        else:
            return make_action_hashable(action) in self.A

    def get_child_node(self, action):
        if is_action_hashable(action):
            return self.children[action]
        else:
            return self.children[make_action_hashable(action)]


class ConvBeltTreeNode(TreeNode):
    def __init__(self, exploration_parameter, depth, state_saver):
        TreeNode.__init__(self, exploration_parameter, depth)
        self.state_saver = state_saver







