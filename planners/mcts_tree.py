from mcts_utils import make_action_hashable, is_action_hashable
import numpy as np


class MCTSTree:
    def __init__(self, root, exploration_parameters):
        self.root = root
        self.nodes = [root]
        print root
        self.exploration_parameters = exploration_parameters

    def has_state(self, state):
        return len([n for n in self.nodes if np.all(n.state == state)]) > 0

    def add_node(self, node, action, parent):
        node.parent = parent
        if is_action_hashable(action):
            parent.children[action] = node
        else:
            parent.children[make_action_hashable(action)] = node

        self.nodes.append(node)

    def is_node_just_added(self, node):
        if node == self.root:
            return False

        for action, resulting_child in zip(node.parent.children.keys(), node.parent.children.values()):
            if resulting_child == node:
                return not (action in node.parent.A)  # action that got to the node is not in parent's actions

    def get_leaf_nodes(self):
        return [n for n in self.nodes if len(n.children.keys()) == 0]

    def get_goal_nodes(self):
        return [n for n in self.nodes if len(n.children.keys()) == 0 and n.is_goal_node]

    def get_best_trajectory_sum_rewards_and_node(self, discount_factor):
        sumR_list = []
        leaf_nodes_for_curr_init_state = []
        leaf_nodes = self.get_leaf_nodes()
        reward_lists = []
        for n in leaf_nodes:
            sumR = 0
            curr_node = n

            # todo take discounting into account
            # collect all rewards into an array
            # dot it with \gamma, gamma^2, ... and so on.
            if n.is_goal_node:
                reward_list = [n.reward_history.values()[0][0]]
            else:
                reward_list = []
            while curr_node.parent is not None:
                reward_list.append(curr_node.parent.reward_history[make_action_hashable(curr_node.parent_action)][0])
                curr_node = curr_node.parent
            reward_lists.append(np.sum(reward_list[::-1]))
            discount_rates = [np.power(discount_factor, i) for i in range(len(reward_list))]
            sumR = np.dot(discount_rates[::-1], reward_list)

            # exclude the ones that are not the descendents of the current init node
            # todo sumR should be -2 on a node that ended with an infeasible action
            sumR_list.append(sumR)
            leaf_nodes_for_curr_init_state.append(n)
        return np.max(sumR_list), leaf_nodes_for_curr_init_state[np.argmax(sumR_list)], max(reward_lists)
