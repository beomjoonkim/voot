import sys
import numpy as np

from mcts import MCTS, MCTSTree
from voronoi_tree_node import VoronoiTreeNode
from mcts_utils import make_action_hashable, is_action_hashable

sys.path.append('../mover_library/')
from samplers import *
from utils import *

DEBUG = True


class VoronoiMCTS(MCTS):
    def __init__(self, pick_pi, place_pi, sampling_strategy, environment):
        dummy_widening_param = -1
        dummy_explr = 1
        MCTS.__init__(self, dummy_widening_param, dummy_explr, pick_pi, place_pi, sampling_strategy,
                      environment)

        # note: grasp lb = [0.1, 0.5, pi/4] ub = [0.9, 1, pi], so radii of inverse reachability dominates
        self.pick_diameter = 0.9811
        self.place_diameter = 2.51*2
        self.s0_node.is_init_node = True
        self.s0_node = VoronoiTreeNode(2*self.max_rwd_to_go(0)/self.pick_diameter, x_diameter=self.pick_diameter, f_star=2.79)
        self.tree = MCTSTree(self.s0_node, self.exploration_parameters)

    def choose_action(self, curr_node):
        is_time_to_sample = curr_node.is_time_to_sample()
        print 'Time to sample new action? ' + str(is_time_to_sample)

        if is_time_to_sample:
            action = self.sample_action(curr_node)
        else:
            action = curr_node.get_best_action()

        return action

    def max_rwd_to_go(self, depth):
        rwd_to_go = 0
        time_steps_to_go = self.depth_limit - depth + 1

        is_pick_time = self.environment.is_pick_time()
        for t in range(time_steps_to_go):
            if is_pick_time:
                if t == 0:
                    reward = 0
                elif t % 2 == 0:
                    reward = 0
                else:
                    reward = 1
            else:
                if t == 0:
                    reward = 1
                elif t % 2 == 0:
                    reward = 1
                else:
                    reward = 0

            rwd_to_go += np.power(self.discount_rate, t)*reward
        return rwd_to_go

    def simulate(self, curr_node, depth):
        if depth == self.depth_limit:
            # arrived at the goal state
            self.found_solution = True
            curr_node.is_goal_node = True
            return

        action = self.choose_action(curr_node)
        if curr_node.is_action_tried(action):
            next_node = curr_node.get_child_node(action)
            if next_node.parent_grasp_or_path is None and not self.environment.is_pick_time():
                do_check_reachability = True
            else:
                do_check_reachability = False
        else:
            do_check_reachability = True
        next_state, reward, grasp_or_path = self.apply_action(action, do_check_reachability)

        if not curr_node.is_action_tried(action):
            if self.environment.is_pick_time():
                next_node = VoronoiTreeNode(2*self.max_rwd_to_go(depth)/self.pick_diameter, x_diameter=self.pick_diameter,
                                            f_star=self.max_rwd_to_go(depth))
            else:
                next_node = VoronoiTreeNode(2*self.max_rwd_to_go(depth)/self.place_diameter, x_diameter=self.place_diameter,
                                            f_star=self.max_rwd_to_go(depth))
            self.tree.add_node(next_node, action, curr_node)

        next_node.parent_action_reward = reward
        next_node.sum_ancestor_action_rewards = curr_node.sum_ancestor_action_rewards + reward

        if next_node.parent_grasp_or_path is None:
            next_node.parent_grasp_or_path = grasp_or_path

        is_infeasible_action = reward == self.environment.infeasible_reward
        if is_infeasible_action:
            # this (s,a) is a dead-end
            next_node.path = None
        else:
            next_node.path = "Exists"
            self.simulate(next_node, depth + 1)

        self.update_node_statistics(curr_node, action, reward)
        return

    def update_node_statistics(self, curr_node, action, reward):
        curr_node.Nvisited += 1
        if is_action_hashable(action):
            hashable_action = action
        else:
            hashable_action = make_action_hashable(action)

        is_action_new = not (hashable_action in curr_node.A)
        if is_action_new:
            curr_node.A.append(hashable_action)
            curr_node.N[hashable_action] = 1
        else:
            curr_node.N[hashable_action] += 1
        # Q =  Q* = r(curr_node,action) + max_a Q(s',a), s' = f(curr_node,action)
        curr_node.Q[hashable_action] = reward + self.discount_rate*curr_node.get_child_max_q(hashable_action)



