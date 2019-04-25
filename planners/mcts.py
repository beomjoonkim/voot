from mcts_tree_node import TreeNode
from mcts_tree import MCTSTree

from generators.uniform import UniformGenerator
from generators.voo import VOOGenerator
from generators.doo import DOOGenerator
from generators.randomized_doo import RandomizedDOOGenerator
from generators.presampled_pick_generator import PreSampledPickGenerator

from mover_library.utils import CustomStateSaver

from generators.gpucb import GPUCBGenerator

import time
import sys
import socket
import numpy as np


from test_scripts.visualize_q_functions import visualize_base_poses_and_q_values

sys.setrecursionlimit(15000)

DEBUG = True

hostname = socket.gethostname()
if hostname == 'dell-XPS-15-9560':
    from mcts_graphics import write_dot_file


class MCTS:
    def __init__(self, widening_parameter, exploration_parameters,
                 sampling_strategy, sampling_strategy_exploration_parameter, c1, n_feasibility_checks,
                 environment, use_progressive_widening, use_ucb, use_max_backup, pick_switch,
                 voo_sampling_mode, voo_counter_ratio, n_switch):
        self.c1 = c1
        self.widening_parameter = widening_parameter
        self.exploration_parameters = exploration_parameters
        self.time_limit = np.inf
        self.environment = environment
        if self.environment.name == 'convbelt':
            self.discount_rate = 0.99  # do we need this?
        else:
            self.discount_rate = 0.9
        self.sampling_strategy = sampling_strategy
        self.sampling_strategy_exploration_parameter = sampling_strategy_exploration_parameter
        self.depth_limit = np.inf
        self.use_progressive_widening = use_progressive_widening
        self.voo_sampling_mode = voo_sampling_mode
        self.voo_counter_ratio = voo_counter_ratio
        self.use_ucb = use_ucb
        self.n_switch = n_switch
        self.n_switch_original = self.n_switch
        self.use_max_backup = use_max_backup
        self.pick_switch = pick_switch

        self.env = self.environment.env
        self.robot = self.environment.robot
        self.s0_node = self.create_node(None, depth=0, reward=0, is_init_node=True)

        self.original_s0_node = self.s0_node
        self.tree = MCTSTree(self.s0_node, self.exploration_parameters)
        self.found_solution = False
        self.goal_reward = 2
        self.infeasible_reward = -2
        self.n_feasibility_checks = n_feasibility_checks

    def create_sampling_agent(self, node, operator_skeleton):
        operator_name = operator_skeleton.type

        if operator_name == 'two_arm_pick' and self.environment.name == 'convbelt':
            return PreSampledPickGenerator()
        if self.sampling_strategy == 'unif':
            return UniformGenerator(operator_name, self.environment)
        elif self.sampling_strategy == 'voo':
            return VOOGenerator(operator_name, self.environment, self.sampling_strategy_exploration_parameter, self.c1,
                                self.voo_sampling_mode, self.voo_counter_ratio)
        elif self.sampling_strategy == 'gpucb':
            return GPUCBGenerator(operator_name, self.environment, self.sampling_strategy_exploration_parameter)
        elif self.sampling_strategy == 'doo':
            return DOOGenerator(node, self.environment, self.sampling_strategy_exploration_parameter)
        elif self.sampling_strategy == 'randomized_doo':
            return RandomizedDOOGenerator(operator_skeleton, self.environment,
                                          self.sampling_strategy_exploration_parameter)
        else:
            print "Wrong sampling strategy"
            return -1

    def create_node(self, parent_action, depth, reward, is_init_node):
        if self.environment.is_goal_reached():
            operator_skeleton = None
        else:
            operator_skeleton = self.environment.get_applicable_op_skeleton(parent_action)

        state_saver = CustomStateSaver(self.environment.env)
        node = TreeNode(operator_skeleton, self.exploration_parameters, depth, state_saver, self.sampling_strategy,
                        is_init_node)
        if not self.environment.is_goal_reached():
            node.sampling_agent = self.create_sampling_agent(node, operator_skeleton)

        node.objects_not_in_goal = self.environment.objects_currently_not_in_goal
        node.parent_action_reward = reward
        node.parent_action = parent_action
        return node

    def retrace_best_plan(self):
        plan = []
        _, _, best_leaf_node = self.tree.get_best_trajectory_sum_rewards_and_node(self.discount_rate)
        curr_node = best_leaf_node

        while not curr_node.parent is None:
            plan.append(curr_node.parent_action)
            curr_node = curr_node.parent

        plan = plan[::-1]
        return plan

    def get_best_goal_node(self):
        leaves = self.tree.get_leaf_nodes()
        goal_nodes = [leaf for leaf in leaves if leaf.is_goal_node]
        if len(goal_nodes) > 1:
            best_traj_reward, curr_node, _ = self.tree.get_best_trajectory_sum_rewards_and_node(self.discount_rate)
        else:
            curr_node = goal_nodes[0]
        return curr_node

    def switch_init_node(self, node):
        self.s0_node.is_init_node = False
        self.s0_node = node
        self.s0_node.is_init_node = True
        self.environment.reset_to_init_state(node)
        self.found_solution = False

    def log_current_tree_to_dot_file(self, iteration):
        if socket.gethostname() == 'dell-XPS-15-9560':
            write_dot_file(self.tree, iteration, '')

    def is_time_to_switch_initial_node(self):
        if self.s0_node.is_goal_node:
            return True

        is_pick_node = self.s0_node.operator_skeleton.type == 'two_arm_pick'

        if len(self.s0_node.Q) == 0:
            n_feasible_actions = 0
        else:
            root_node_reward_history = self.s0_node.reward_history.values()
            root_node_reward_history = np.array([np.max(R) for R in root_node_reward_history])
            n_feasible_actions = np.sum(root_node_reward_history >= 0)

        if is_pick_node:
            if self.pick_switch:
                we_evaluated_the_node_enough = n_feasible_actions >= self.n_switch
            else:
                we_evaluated_the_node_enough = n_feasible_actions > 0
        else:
            we_evaluated_the_node_enough = n_feasible_actions >= self.n_switch

        return we_evaluated_the_node_enough

    def choose_child_node_to_descend_to(self):
        is_child_goal_node = np.any([c.is_goal_node for c in self.s0_node.children.values()])
        if is_child_goal_node:
            best_node = self.tree.root
            self.n_switch += self.n_switch
        else:
            non_goal_traj_feasible_actions = []
            non_goal_traj_q_values = []
            is_pick_node = self.s0_node.operator_skeleton.type == 'two_arm_pick'
            for a in self.s0_node.A:
                is_action_feasible = np.max(self.s0_node.reward_history[a]) > self.infeasible_reward

                if is_pick_node and not self.pick_switch:
                    is_action_evaled_enough = False  # free-pass if we are not doing pick switch
                else:
                    is_action_evaled_enough = self.s0_node.children[a].have_been_used_as_root
                """
                if is_pick_node and not self.pick_switch:
                    is_action_evaled_enough = False  # free-pass if we are not doing pick switch
                else:
                    is_action_evaled_enough = self.s0_node.children[a].get_n_feasible_actions(self.infeasible_reward) \
                                              >= self.n_switch_original
                """

                if is_action_feasible and (not is_action_evaled_enough):
                    non_goal_traj_feasible_actions.append(a)
                    non_goal_traj_q_values.append(a)
            best_action = non_goal_traj_feasible_actions[np.argmax(non_goal_traj_q_values)]
            best_node = self.s0_node.children[best_action]
            best_node.have_been_used_as_root = True
        return best_node

    def search(self, n_iter=100, max_time=np.inf):
        depth = 0
        time_to_search = 0
        search_time_to_reward = []
        plan = None

        self.n_iter = n_iter
        for iteration in range(n_iter):
            print '*****SIMULATION ITERATION %d' % iteration
            print '*****Root node idx %d' % self.s0_node.idx
            self.environment.reset_to_init_state(self.s0_node)

            if self.is_time_to_switch_initial_node():
                print "Switching root node!"
                if self.s0_node.A[0].type == 'two_arm_place':
                    self.s0_node.store_node_information(self.environment.name)
                    #visualize_base_poses_and_q_values(self.s0_node.Q, self.environment)
                best_child_node = self.choose_child_node_to_descend_to()
                self.switch_init_node(best_child_node)

            stime = time.time()
            self.simulate(self.s0_node, depth)
            time_to_search += time.time() - stime

            #self.log_current_tree_to_dot_file(iteration)
            best_traj_rwd, progress, best_node = self.tree.get_best_trajectory_sum_rewards_and_node(self.discount_rate)
            search_time_to_reward.append([time_to_search, iteration, best_traj_rwd, len(progress)])
            plan = self.retrace_best_plan()
            rewards = np.array([np.max(rlist) for rlist in self.s0_node.reward_history.values()])
            print 'n feasible actions , n_switch ', np.sum(rewards >= 0), self.n_switch
            print search_time_to_reward[-1], np.argmax(np.array(search_time_to_reward)[:, 2])

            if time_to_search > max_time:
                break

        self.environment.reset_to_init_state(self.s0_node)
        return search_time_to_reward, self.s0_node.best_v, plan

    def choose_action(self, curr_node, depth):
        print "Widening parameter ", self.widening_parameter*np.power(0.8, depth)
        if not curr_node.is_reevaluation_step(self.widening_parameter*np.power(0.8, depth), self.environment.infeasible_reward,
                                              self.use_progressive_widening, self.use_ucb):
            print "Is time to sample new action? True"
            new_continuous_parameters = self.sample_continuous_parameters(curr_node)
            curr_node.add_actions(new_continuous_parameters)
            action = curr_node.A[-1]
        else:
            print "Re-evaluation? True"
            if self.use_ucb:
                action = curr_node.perform_ucb_over_actions()
            else:
                action = curr_node.choose_new_arm()
        return action

    def update_node_statistics(self, curr_node, action, sum_rewards, reward):
        # todo rewrite this function
        curr_node.Nvisited += 1

        is_action_never_tried = curr_node.N[action] == 0
        if is_action_never_tried:
            curr_node.reward_history[action] = [reward]
            curr_node.N[action] += 1
            curr_node.Q[action] = sum_rewards
        else:
            curr_node.reward_history[action].append(reward)
            curr_node.N[action] += 1
            if self.use_max_backup and sum_rewards > curr_node.Q[action]:
                curr_node.Q[action] = sum_rewards
            else:
                curr_node.Q[action] += (sum_rewards - curr_node.Q[action]) / float(curr_node.N[action])

    @staticmethod
    def update_goal_node_statistics(curr_node, reward):
        # todo rewrite this function
        curr_node.Nvisited += 1
        curr_node.reward = reward

    def visualize_samples_from_sampling_agent(self, node):
        action, status, doo_node, action_parameters = node.sampling_agent.sample_feasible_action(node,
                                                                                                 self.n_feasibility_checks)

    def simulate(self, curr_node, depth):
        if self.environment.is_goal_reached():
            # arrived at the goal state
            if not curr_node.is_goal_and_already_visited:
                # todo mark the goal trajectory, and don't revisit the actions on the goal trajectory
                self.found_solution = True
                curr_node.is_goal_node = True
                print "Solution found, returning the goal reward", self.goal_reward
                self.update_goal_node_statistics(curr_node, self.goal_reward)
            return self.goal_reward

        if depth == self.depth_limit:
            return 0

        if DEBUG:
            print "At depth ", depth
            print "Is it time to pick?", self.environment.is_pick_time()

        action = self.choose_action(curr_node, depth)
        reward = self.environment.apply_operator_instance(action, curr_node)
        print "Executed ", action.type, action.continuous_parameters['is_feasible'], action.discrete_parameters
        print "reward ", reward
        if action.type.find('paps') != -1 and reward > -2:
            import pdb;pdb.set_trace()

        # for the same action, we now get different results because RRT calls to the same goal would result in
        # both feasible and infeasible.

        if not curr_node.is_action_tried(action):
            next_node = self.create_node(action, depth+1, reward, is_init_node=False)
            self.tree.add_node(next_node, action, curr_node)
            next_node.sum_ancestor_action_rewards = next_node.parent.sum_ancestor_action_rewards + reward
        else:
            next_node = curr_node.children[action]

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

        parent_reward_to_node = node.reward_history[action][0]
        parent_sum_rewards = parent_reward_to_node + self.discount_rate * child_sum_rewards
        self.update_node_statistics(node, action, parent_sum_rewards, parent_reward_to_node)
        self.update_ancestor_node_statistics(node.parent, node.parent_action, parent_sum_rewards)

    def sample_continuous_parameters(self, node):
        return node.sampling_agent.sample_next_point(node, self.n_feasibility_checks)


