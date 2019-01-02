import sys
sys.path.append('../mover_library/')

from samplers import *
from utils import *
import socket

from mcts import MCTS

if socket.gethostname() == 'dell-XPS-15-9560':
    from mcts_graphics import write_dot_file

import time
import numpy as np

DEBUG = True


class ConstrainedMCTS(MCTS):
    def __init__(self, widening_parameter, exploration_parameters,
                 sampling_strategy, environment, domain_name, task_plan, constraints):
        MCTS.__init__(self, widening_parameter, exploration_parameters, sampling_strategy, environment, domain_name, task_plan)
        self.constraints = constraints
        self.constraint_idx = 0

    def enforce_constraints(self, curr_node):
        if curr_node.is_init_node:
            self.constraint_idx = 0

        pick_constraints = self.constraints  # currently only support initial pick constraints for packing
        is_pick_node = self.environment.is_pick_time()
        const_obj_name = pick_constraints[self.constraint_idx]['obj_name']
        pick_config = pick_constraints[self.constraint_idx]['pick_config']
        pick_base_config = pick_constraints[self.constraint_idx]['pick_base_config']

        if const_obj_name == curr_node.obj.GetName() and is_pick_node:
            self.environment.apply_pick_constraint(const_obj_name, pick_config, pick_base_config)
            self.environment.env.GetKinBody(const_obj_name).Enable(True)

            if self.constraint_idx < len(self.constraints):
                self.environment.fetch_base_config = pick_constraints[self.constraint_idx]['pick_base_config']
            else:
                self.environment.fetch_base_config = None

            curr_node.operator = self.environment.which_operator(curr_node.obj)
            self.constraint_idx += 1
            curr_node.region = self.task_plan[self.task_plan_idx]['region']

    def simulate(self, curr_node, depth):
        if self.is_goal_reached():
            # arrived at the goal state
            self.found_solution = True
            print "Solution found"
            curr_node.is_goal_node = True
            return 5

        self.enforce_constraints(curr_node)

        if DEBUG:
            print "At depth ", depth
            print "Is it time to pick?", self.environment.is_pick_time()

        action = self.choose_action(curr_node)
        print action, curr_node.operator
        parent_motion = None
        if curr_node.is_action_tried(action):
            if DEBUG:
                print "Executing tree policy, taking action ", action
            next_node = curr_node.get_child_node(action)
            if next_node.parent_motion is None:
                check_feasibility = True # todo put this back
            else:
                parent_motion = next_node.parent_motion
                check_feasibility = False
        else:
            check_feasibility = True  # todo: store path

        if DEBUG:
            print 'Is pick time? ', self.environment.is_pick_time()
            print "Executing action ", action

        #if self.environment.is_solving_namo and curr_node.obj.GetName()=='rectangular_packing_box3':
        #    import pdb;pdb.set_trace()
        next_state, reward, parent_motion = self.apply_action(curr_node, action, check_feasibility, parent_motion)
        print 'Reward ', reward
        #if self.environment.is_solving_namo:
        #    import pdb;pdb.set_trace()

        self.update_task_plan_indices(reward) # create the next node based on the updated task plan progress

        if not curr_node.is_action_tried(action):
            next_node = self.create_node(depth+1, reward, is_init_node=False)
            self.tree.add_node(next_node, action, curr_node)
            next_node.sum_ancestor_action_rewards = next_node.parent.sum_ancestor_action_rewards + reward

        if next_node.parent_motion is None and reward != self.environment.infeasible_reward:
            #if DEBUG:
            #    print "Updating parent_motion", parent_motion
            next_node.parent_motion = parent_motion

        is_infeasible_action = reward == self.environment.infeasible_reward
        if is_infeasible_action:
            # this (s,a) is a dead-end
            sum_rewards = reward
        else:
            sum_rewards = reward + self.discount_rate * self.simulate(next_node, depth + 1)

        self.update_node_statistics(curr_node, action, sum_rewards)
        return sum_rewards



