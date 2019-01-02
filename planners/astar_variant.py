import sys

from sampling_strategies.voo import VOO
#from sampling_strategies.voo import AnalyticalVOO
from sampling_strategies.uniform import Uniform

sys.path.append('../mover_library/')
from mcts_tree_node import ConvBeltTreeNode
from samplers import *
from utils import *
from mcts_utils import make_action_hashable, is_action_hashable
from utils import draw_robot_at_conf, remove_drawn_configs
import socket

from mcts_graphics import write_dot_file

import time
import numpy as np
from misc.priority_queue import PriorityQueue

DEBUG = True


class MCTSTree:
    def __init__(self, root, exploration_parameters):
        self.root = root
        self.nodes = [root]
        self.exploration_parameters = exploration_parameters

    def has_state(self, state):
        return len([n for n in self.nodes if np.all(n.state == state)]) > 0

    def add_node(self, node, action, parent):
        node.parent = parent
        if is_action_hashable(action):
            parent.children[action] = node
        else:
            parent.children[make_action_hashable(action)] = node
        node.parent_action = action
        self.nodes.append(node)

    def is_node_just_added(self, node):
        if node == self.root:
            return False

        for action, resulting_child in zip(node.parent.children.keys(), node.parent.children.values()):
            if resulting_child == node:
                return not (action in node.parent.A)  # action that got to the node is not in parent's actions

    def get_leaf_nodes(self):
        return [n for n in self.nodes if len(n.children.keys()) == 0]

    def get_best_trajectory_sum_rewards(self):
        return max([n.sum_ancestor_action_rewards for n in self.get_leaf_nodes()])


class AStarVariant:
    def __init__(self, pick_pi, place_pi, sampling_strategy, environment):
        self.time_limit = np.inf
        self.depth_limit = 8
        self.discount_rate = 0.9

        self.pick_pi = pick_pi
        self.place_pi = place_pi
        self.environment = environment
        self.s0_node = ConvBeltTreeNode(0, 0, self.environment.get_state_saver())
        self.s0_node.is_init_node = True
        self.found_solution = False
        self.tree_root = self.s0_node

        if DEBUG:
            self.environment.env.SetViewer('qtcoin')

        if sampling_strategy == 'voo':
            self.sampling_strategy = VOO(self.environment, self.pick_pi, self.place_pi, explr_p=0.3)
        else:
            self.sampling_strategy = Uniform(self.environment, self.pick_pi, self.place_pi)

    def search(self):
        max_rewards = 4.0591
        queue = PriorityQueue()
        queue.push(0, (0, self.s0_node))

        for iter in range(100):
            print '*****SIMULATION ITERATION %d' % iter
            priority, curr_node = queue.pop()
            curr_node_depth = curr_node.depth

            self.environment.restore(curr_node.state_saver)
            # todo I think I am adding children to the wrong parent?
            action = self.sample_action(curr_node, curr_node.state_saver.is_pick_node)

            # compute the next node
            next_state, reward, grasp_or_path = self.apply_action(action, True)
            next_node = ConvBeltTreeNode(0,
                                         curr_node_depth+1,
                                         self.environment.get_state_saver())
            is_infeasible_action = reward == self.environment.infeasible_reward
            if is_infeasible_action:
                # this (s,a) is a dead-end
                next_node.path = None
                sum_rewards = reward
            else:
                next_node.path = "Exists"
                sum_rewards = curr_node.sum_rewards + np.power(self.discount_rate, curr_node_depth+1) * reward

            next_node.sum_rewards = sum_rewards
            next_node.parent_grasp_or_path = grasp_or_path
            next_node.parent = curr_node
            self.update_node_statistics(next_node, action, sum_rewards)

            # save the current trajectory
            if curr_node.parent is not None:
                queue.push(-curr_node.parent.Q[curr_node.parent_action],
                           (-curr_node.parent.Q[curr_node.parent_action], curr_node))
            else:
                queue.push(np.inf, (np.inf, curr_node))
            queue.push(-sum_rewards, (-sum_rewards, next_node))
            print queue.queue
            import pdb; pdb.set_trace()

    @staticmethod
    def update_node_statistics(new_node, action, sum_rewards):
        hashable_action = make_action_hashable(action)
        parent_node = new_node.parent
        curr_node = new_node
        parent_node.A.append(hashable_action)
        parent_node.Q[hashable_action] = sum_rewards  # only one child
        parent_node.children[hashable_action] = new_node
        while parent_node is not None:
            if hashable_action not in parent_node.A:
                import pdb;pdb.set_trace()
                best_child_Q_val = np.max(curr_node.Q.values())
                parent_node.Q[curr_node.parent_action] = best_child_Q_val  # parent_action is not being updated
            curr_node = curr_node.parent
            parent_node = curr_node.parent

    def apply_action(self, action, do_check_reachability):
        if action is None:
            return None, self.environment.infeasible_reward, None

        if self.environment.is_pick_time():
            next_state, reward, grasp = self.environment.apply_pick_action(action)
            return next_state, reward, grasp
        else:
            next_state, reward, path = self.environment.apply_place_action(action, do_check_reachability)
            return next_state, reward, path

    def sample_action(self, node, is_pick_node):
        action = self.sampling_strategy.sample_next_point(node, is_pick_node)
        return action


