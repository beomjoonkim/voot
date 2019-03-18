import sys
import socket
import numpy as np

from mcts_tree_node import TreeNode
from mcts_tree import MCTSTree
from mcts_utils import make_action_hashable, is_action_hashable

from generators.uniform import UniformGenerator
from generators.voo import VOOGenerator
from generators.doo import DOOGenerator
from generators.randomized_doo import RandomizedDOOGenerator

## openrave helper libraries
sys.path.append('../mover_library/')
from manipulation.primitives.savers import DynamicEnvironmentStateSaver
from samplers import *
from utils import *

import time
from generators.doo_utils.doo_tree import BinaryDOOTree, DOOTreeNode
sys.path.append('../mover_library/')
from utils import get_pick_domain, get_place_domain
from generators.gpucb import GPUCBGenerator

sys.setrecursionlimit(15000)

DEBUG = True

hostname = socket.gethostname()
if hostname == 'dell-XPS-15-9560':
    from mcts_graphics import write_dot_file


def create_doo_agent(operator):
    if operator == 'two_arm_pick':
        domain = get_pick_domain()
    else:
        domain = get_place_domain()
    return BinaryDOOTree(domain)


class MCTS:
    def __init__(self, widening_parameter, exploration_parameters,
                 sampling_strategy, sampling_strategy_exploration_parameter, c1, n_feasibility_checks,
                 environment, domain_name):
        self.c1 = c1
        self.progressive_widening_parameter = widening_parameter
        self.exploration_parameters = exploration_parameters
        self.time_limit = np.inf
        if domain_name == 'namo':
            self.discount_rate = 0.9
        else:
            self.discount_rate = 1
        self.environment = environment
        self.sampling_strategy = sampling_strategy
        self.sampling_strategy_exploration_parameter = sampling_strategy_exploration_parameter
        self.depth_limit = 300

        self.env = self.environment.env
        self.robot = self.environment.robot
        self.s0_node = self.create_node(None, depth=0, reward=0, is_init_node=True)

        self.original_s0_node = self.s0_node
        self.tree = MCTSTree(self.s0_node, self.exploration_parameters)
        self.found_solution = False
        if self.environment.name == 'namo':
            self.goal_reward = 2
        else:
            self.goal_reward = 0
        self.n_feasibility_checks = n_feasibility_checks

        """
        if domain_name == 'convbelt':
            self.depth_limit = 10
            self.is_satisficing_problem = True
        elif domain_name == 'namo':
            self.depth_limit = np.inf
            self.is_satisficing_problem = False
        """

    def create_sampling_agent(self, node, operator_name):
        if self.sampling_strategy == 'unif':
            return UniformGenerator(operator_name, self.environment)
        elif self.sampling_strategy == 'voo':
            return VOOGenerator(operator_name, self.environment, self.sampling_strategy_exploration_parameter, self.c1)
        elif self.sampling_strategy == 'gpucb':
            return GPUCBGenerator(operator_name, self.environment, self.sampling_strategy_exploration_parameter)
        elif self.sampling_strategy == 'doo':
            return DOOGenerator(node, self.environment, self.sampling_strategy_exploration_parameter)
        elif self.sampling_strategy == 'randomized_doo':
            return RandomizedDOOGenerator(node, self.environment, self.sampling_strategy_exploration_parameter)
        else:
            print "Wrong sampling strategy"
            return -1

    def create_node(self, parent_action, depth, reward, is_init_node):
        if self.environment.is_goal_reached():
            operator_skeleton = None
        else:
            operator_skeleton = self.environment.get_applicable_op()

        state_saver = DynamicEnvironmentStateSaver(self.environment.env)
        node = TreeNode(operator_skeleton, self.exploration_parameters, depth, state_saver,
                        self.sampling_strategy, is_init_node)

        if not self.environment.is_goal_reached():
            node.sampling_agent = self.create_sampling_agent(node, operator_skeleton.type)

        node.objects_not_in_goal = self.environment.objects_not_in_goal
        node.parent_action_reward = reward
        node.parent_action = parent_action
        return node

    @staticmethod
    def get_best_child_node(node):
        if len(node.children) == 0:
            return None
        else:
            # returns the most visited chlid
            # another option is to return the child with best value
            best_child_action_idx = np.argmax(node.N.values())
            best_child_action = node.N.keys()[best_child_action_idx]
            return node.children[best_child_action]

    def retrace_best_plan(self, best_node):
        plan = []
        #curr_node = self.get_best_goal_node()
        curr_node = best_node
        #while not curr_node.is_init_node:
        while not curr_node.parent is None:
            action = curr_node.parent_action
            path = curr_node.parent_motion
            obj = curr_node.parent.obj
            obj_name = obj.GetName()
            operator = curr_node.parent.operator
            objs_in_collision = curr_node.objs_in_collision
            plan.insert(0, {'action': action, 'path': path, 'obj_name': obj_name, 'operator': operator,
                            'obj_names_in_collision': [obj.GetName() for obj in objs_in_collision]})
            curr_node = curr_node.parent
        return plan

    def get_best_goal_node(self):
        leaves = self.tree.get_leaf_nodes()
        goal_nodes = [leaf for leaf in leaves if leaf.is_goal_node]
        if len(goal_nodes) > 1:
            best_traj_reward, curr_node,_  = self.tree.get_best_trajectory_sum_rewards_and_node(self.discount_rate)
            #curr_node = [leaf for leaf in goal_nodes if leaf.sum_ancestor_action_rewards == best_traj_reward][0]
        else:
            curr_node = goal_nodes[0]
        return curr_node

    def switch_init_node(self, node):
        self.environment.reset_to_init_state(node)
        self.s0_node.is_init_node = False

        self.s0_node = node
        self.s0_node.is_init_node = True
        self.found_solution = False

    def search(self, n_iter=100, n_optimal_iter=0, max_time=np.inf):
        # n_optimal_iter: additional number of iterations you are allowed to run after finding a solution
        depth = 0
        time_to_search = 0
        search_time_to_reward = []
        optimal_iter = 0
        n_node_switch = 0
        switch_counter = 0
        found_solution_permanent = False
        reward_lists = []
        for iteration in range(n_iter):
            print '*****SIMULATION ITERATION %d' % iteration
            self.environment.reset_to_init_state(self.s0_node)

            stime = time.time()
            self.simulate(self.s0_node, depth)
            time_to_search += time.time() - stime

            best_traj_rwd, best_node, reward_list = self.tree.get_best_trajectory_sum_rewards_and_node(self.discount_rate)
            search_time_to_reward.append([time_to_search, iteration, best_traj_rwd,  self.found_solution])
            reward_lists.append(reward_list)
            plan = [self.retrace_best_plan(best_node), best_traj_rwd, self.found_solution]

            goal_node = None

            if time_to_search > max_time:
                break

        self.environment.reset_to_init_state(self.s0_node)
        return search_time_to_reward, plan, goal_node, reward_lists

    def choose_action(self, curr_node):
        n_actions = len(curr_node.A)
        is_time_to_sample = n_actions <= self.progressive_widening_parameter * curr_node.Nvisited
        if len(curr_node.Q.values()) > 0:
            best_Q = np.max(curr_node.Q.values())
            is_next_node_goal = np.all([child.is_goal_node for child in curr_node.children.values()])
            is_time_to_sample = is_time_to_sample or (best_Q == self.environment.infeasible_reward) or is_next_node_goal

        if is_time_to_sample:
            new_continuous_parameters = self.sample_continuous_parameters(curr_node)
            curr_node.add_actions(new_continuous_parameters)

        action = curr_node.perform_ucb_over_actions()

        return action

    @staticmethod
    def update_node_statistics(curr_node, action, sum_rewards, reward):
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
            curr_node.reward_history[hashable_action] = [reward]
        else:
            curr_node.N[hashable_action] += 1
            curr_node.reward_history[hashable_action].append(reward)
            curr_node.Q[hashable_action] += (sum_rewards - curr_node.Q[hashable_action]) / \
                                            float(curr_node.N[hashable_action])

    def simulate(self, curr_node, depth):
        if self.environment.is_goal_reached():
            # arrived at the goal state
            if not curr_node.is_goal_and_already_visited:
                self.found_solution = True
                curr_node.is_goal_node = True
                print "Solution found, returning the goal reward", self.goal_reward
                self.update_node_statistics(curr_node, curr_node.parent_action, self.goal_reward, self.goal_reward)
            return self.goal_reward

        if depth == self.depth_limit:
            return 0

        if DEBUG:
            print "At depth ", depth
            print "Is it time to pick?", self.environment.is_pick_time()

        action = self.choose_action(curr_node)
        reward = self.environment.apply_action(curr_node, action)

        if not curr_node.is_action_tried(action):
            next_node = self.create_node(action, depth+1, reward, is_init_node=False)
            self.tree.add_node(next_node, action, curr_node)
            next_node.sum_ancestor_action_rewards = next_node.parent.sum_ancestor_action_rewards + reward

        is_infeasible_action = reward == self.environment.infeasible_reward
        if is_infeasible_action:
            sum_rewards = reward
        else:
            sum_rewards = reward + self.discount_rate * self.simulate(next_node, depth + 1)

        self.update_node_statistics(curr_node, action, sum_rewards, reward)
        if curr_node.is_init_node and curr_node.parent is not None:
            self.update_ancestor_node_statistics(curr_node.parent, curr_node.parent_action, sum_rewards)

        return sum_rewards

    def update_ancestor_node_statistics(self, node, action, child_sum_rewards):
        if node is None:
            return

        parent_reward_to_node = node.reward_history[make_action_hashable(action)][0]
        parent_sum_rewards = parent_reward_to_node + self.discount_rate * child_sum_rewards
        self.update_node_statistics(node, action, parent_sum_rewards, parent_reward_to_node)

        self.update_ancestor_node_statistics(node.parent, node.parent_action, parent_sum_rewards)

    """
    def apply_action(self, node, action, check_feasibility, parent_motion):
        if action is None:
            return None, self.environment.infeasible_reward, None
        which_operator = self.environment.which_operator(node.obj)
        if which_operator == 'two_arm_pick':
            if self.environment.name == 'convbelt':
                self.environment.disable_objects()
                node.obj.Enable(True)
            next_state, reward, path, objs_in_collision = self.environment.apply_two_arm_pick_action(action, node, check_feasibility, parent_motion)
            if self.environment.name == 'convbelt':
                self.environment.enable_objects()
                node.obj.Enable(True)
        elif which_operator == 'two_arm_place':
            next_state, reward, path, objs_in_collision = self.environment.apply_two_arm_place_action(action, node, check_feasibility, parent_motion)
        elif which_operator == 'one_arm_pick':
            next_state, reward, path, objs_in_collision = self.environment.apply_one_arm_pick_action(action, node.obj, node.region, check_feasibility, parent_motion)
        elif which_operator == 'one_arm_place':
            next_state, reward, path, objs_in_collision = self.environment.apply_one_arm_place_action(action, node.obj, node.region, check_feasibility, parent_motion)
        elif which_operator == 'next_base_pose':
            next_state, reward, path, objs_in_collision = self.environment.apply_next_base_pose(action, node, check_feasibility, parent_motion)

        return next_state, reward, path, objs_in_collision
    """

    def sample_continuous_parameters(self, node):
        return node.sampling_agent.sample_next_point(node, self.n_feasibility_checks)


