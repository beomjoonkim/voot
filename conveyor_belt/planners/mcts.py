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


class MCTS:
    def __init__(self, widening_parameter, exploration_parameters, pick_pi, place_pi,
                 sampling_strategy, environment):
        self.progressive_widening_parameter = widening_parameter
        self.exploration_parameters = exploration_parameters

        self.time_limit = np.inf
        self.depth_limit = 10
        self.discount_rate = 0.9

        self.pick_pi = pick_pi
        self.place_pi = place_pi
        self.environment = environment
        self.s0_node = ConvBeltTreeNode(self.exploration_parameters, 0, self.environment.get_state_saver())
        self.s0_node.is_init_node = True
        self.tree = MCTSTree(self.s0_node, self.exploration_parameters)
        self.found_solution = False

        if DEBUG:
            self.environment.env.SetViewer('qtcoin')
        if sampling_strategy == 'voo':
            self.sampling_strategy = VOO(self.environment, self.pick_pi, self.place_pi, explr_p=0.3)
        else:
            self.sampling_strategy = Uniform(self.environment, self.pick_pi, self.place_pi)

    @staticmethod
    def get_best_child_node(node):
        if len(node.children) == 0:
            return None
        else:
            #best_child_action_idx = np.argmax(node.Q.values)
            #best_child_action = node.Q.keys()[best_child_action_idx]
            best_child_action_idx = np.argmax(node.N.values())
            best_child_action = node.N.keys()[best_child_action_idx]
            return node.children[best_child_action]

    def retrace_plan(self):
        plan = []
        leaves = self.tree.get_leaf_nodes()
        curr_node = [leaf for leaf in leaves if leaf.is_goal_node][0]
        while curr_node.parent_grasp_or_path is not None:
            action = (curr_node.parent_action, curr_node.parent_grasp_or_path)
            plan.insert(0, action)
            curr_node = curr_node.parent
        return plan

    def switch_init_node(self, node):
        self.environment.set_init_state(node.state_saver)
        self.environment.reset_to_init_state()
        self.s0_node.is_init_node = False
        self.s0_node = node
        self.s0_node.is_init_node = True

    def search(self):
        depth = 0
        i = 0
        time_to_search = 0
        search_time_to_reward = []
        for iter in range(100):
            if self.s0_node.Nvisited >= 5:
                best_child = self.get_best_child_node(self.s0_node)
                is_best_child_infeasible = best_child.parent_grasp_or_path is None
                if is_best_child_infeasible and self.s0_node.parent is not None:
                    print "Switching the initial node failed, back-tracking to parent"
                    self.switch_init_node(self.s0_node.parent)
                    depth -= 1
                else:
                    print "Switching the initial node to the best child"
                    self.switch_init_node(best_child)
                    depth += 1
            print '*****SIMULATION ITERATION %d' % iter
            self.environment.reset_to_init_state()
            stime = time.time()
            self.simulate(self.s0_node, depth)
            time_to_search += time.time() - stime
            if socket.gethostname() == 'dell-XPS-15-9560':
                write_dot_file(self.tree, i)
            best_traj_rwd = self.tree.get_best_trajectory_sum_rewards()
            search_time_to_reward.append([time_to_search, best_traj_rwd])
            i += 1
            if self.found_solution is True:
                plan = self.retrace_plan()
                break
            else:
                plan = None

        return search_time_to_reward, plan

    def choose_action(self, curr_node):
        print 'N(A), progressive parameter = %d,%f' % (len(curr_node.A),
                                                       self.progressive_widening_parameter * curr_node.Nvisited)

        n_actions = len(curr_node.A)
        is_time_to_sample = n_actions <= self.progressive_widening_parameter * curr_node.Nvisited
        if len(curr_node.Q.values()) > 0:
            best_Q = np.max(curr_node.Q.values())
            is_time_to_sample = is_time_to_sample or (best_Q == self.environment.infeasible_reward)

        print 'Time to sample new action? ' + str(is_time_to_sample)

        if is_time_to_sample:
            action = self.sample_action(curr_node, curr_node.state_saver.is_pick_node)
        else:
            action = curr_node.get_best_action()

        return action

    @staticmethod
    def update_node_statistics(curr_node, action, sum_rewards):
        curr_node.Nvisited += 1
        if is_action_hashable(action):
            hashable_action = action
        else:
            hashable_action = make_action_hashable(action)
        is_action_new = not (hashable_action in curr_node.A)
        if is_action_new:
            curr_node.A.append(hashable_action)
            curr_node.N[hashable_action] = 1
            curr_node.Q[hashable_action] = sum_rewards
            curr_node.sum_rewards_history[hashable_action] = [sum_rewards]
        else:
            curr_node.N[hashable_action] += 1
            curr_node.sum_rewards_history[hashable_action].append(sum_rewards)
            curr_node.Q[hashable_action] += (sum_rewards - curr_node.Q[hashable_action]) / \
                                            float(curr_node.N[hashable_action])

    def simulate(self, curr_node, depth):
        if depth == self.depth_limit:
            # arrived at the goal state
            self.found_solution = True
            print "Solution found"
            curr_node.is_goal_node = True
            return 0

        print "At depth ", depth
        print "Is it time to pick?", self.environment.is_pick_time()
        action = self.choose_action(curr_node)
        if curr_node.is_action_tried(action):
            print "Executing tree policy, taking action ", action
            next_node = curr_node.get_child_node(action)
            if next_node.parent_grasp_or_path is None and not self.environment.is_pick_time():
                do_check_reachability = True # todo put this back
                #do_check_reachability = False
            else:
                do_check_reachability = False
        else:
            print "(n,a) was never tried. Expand with new action ", action
            do_check_reachability = True  # todo: store path
            #do_check_reachability = False

        print 'Is pick time? ', self.environment.is_pick_time()
        #if DEBUG:
        #    if not self.environment.is_pick_time():
        #        draw_robot_at_conf(action, 0, 'cg', self.environment.robot, self.environment.env)
        #    import pdb;pdb.set_trace()
        #    remove_drawn_configs('cg', self.environment.env)
        print "Executing action ", action
        next_state, reward, grasp_or_path = self.apply_action(action, do_check_reachability)
        print 'Reward ', reward

        if not curr_node.is_action_tried(action):
            next_node = ConvBeltTreeNode(self.exploration_parameters, depth+1, self.environment.get_state_saver())
            self.tree.add_node(next_node, action, curr_node)

        next_node.parent_action_reward = reward
        next_node.sum_ancestor_action_rewards = curr_node.sum_ancestor_action_rewards + reward

        if next_node.parent_grasp_or_path is None:
            next_node.parent_grasp_or_path = grasp_or_path

        is_infeasible_action = reward == self.environment.infeasible_reward
        if is_infeasible_action:
            # this (s,a) is a dead-end
            next_node.path = None
            sum_rewards = reward
        else:
            next_node.path = "Exists"
            sum_rewards = reward + self.discount_rate * self.simulate(next_node, depth + 1)

        self.update_node_statistics(curr_node, action, sum_rewards)
        return sum_rewards

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


