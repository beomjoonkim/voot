import sys
import numpy as np

sys.path.append('../mover_library/')
from generator import Generator
from planners.mcts_utils import make_action_executable

from utils import pick_parameter_distance, place_parameter_distance
from doo_utils.doo_tree import BinaryDOOTree
from utils import pick_parameter_distance, place_parameter_distance

import matplotlib.pyplot as plt
import copy


class DOOGenerator(Generator):
    def __init__(self, node, problem_env, explr_p):
        operator_name = node.operator
        Generator.__init__(self, operator_name, problem_env)
        self.explr_p = explr_p
        self.x_min = copy.deepcopy(self.domain[0])
        self.x_max = copy.deepcopy(self.domain[1])
        self.domain[0] = self.normalize_x_value(self.domain[0])  # (self.domain[0] - self.x_min) / (self.x_max-self.x_min)
        self.domain[1] = self.normalize_x_value(self.domain[1])  # (self.domain[1] - self.x_min) / (self.x_max-self.x_min)

        if operator_name == 'two_arm_pick':
            pick_param_distance_for_obj = lambda x,y: pick_parameter_distance(node.obj, x, y)
            #euclidean_dist = lambda x,y: np.linalg.norm(x-y)
            self.doo_tree = BinaryDOOTree(self.domain, self.explr_p, pick_param_distance_for_obj)  # this depends on the problem
        elif operator_name == 'two_arm_place':
            #euclidean_dist = lambda x,y: np.linalg.norm(x-y)
            place_dist = lambda x,y: place_parameter_distance(x, y, 1)
            self.doo_tree = BinaryDOOTree(self.domain, self.explr_p, place_dist)  # this depends on the problem
        else:
            print "Wrong operator name"
            sys.exit(-1)
        self.update_flag = 'update_me'

    def sample_next_point(self, node, n_iter):
        self.update_evaled_values(node)

        normalized_evaled_actions = [self.normalize_x_value(a) for a in self.evaled_actions]
        self.doo_tree.update_evaled_values(normalized_evaled_actions, self.evaled_q_values, self.problem_env.infeasible_reward)
        print "DOO sampling..."

        for i in range(n_iter):
            action_parameters, doo_node = self.choose_next_point()
            action, status = self.feasibility_checker.check_feasibility(node, action_parameters)
            if status == 'HasSolution':
                self.evaled_actions.append(action_parameters)
                self.doo_tree.expand_node(self.update_flag, doo_node)
                self.evaled_q_values.append(self.update_flag)
                print "Found feasible sample"
                break
            else:
                #self.evaled_q_values.append(self.problem_env.infeasible_reward)
                #self.doo_tree.expand_node(self.problem_env.infeasible_reward, doo_node)
                pass

        return action

    def choose_next_point(self):
        next_node = self.doo_tree.get_next_point_and_node_to_evaluate()
        next_node.evaluated_x = next_node.cell_mid_point
        x_to_evaluate = next_node.cell_mid_point
        x_to_evaluate = self.unnormalize_x_value(x_to_evaluate)
        return x_to_evaluate, next_node

    def unnormalize_x_value(self, x_value):
        return x_value
        return x_value * (self.x_max - self.x_min) + self.x_min

    def normalize_x_value(self, x_value):
        return x_value
        return (x_value - self.x_min) / (self.x_max - self.x_min)

def main():
    domain = np.array([[-10, -10], [10, 10]])
    doo_tree = BinaryDOOTree(domain)

    target_fcn = lambda x, y: -(x**2+y**2)

    plt.figure()
    evaled_points = []
    for i in range(100):
        next_node = doo_tree.get_next_node_to_evaluate()
        x_to_evaluate = next_node.x_value
        fval = target_fcn(x_to_evaluate[0], x_to_evaluate[1])
        doo_tree.expand_node(fval, next_node)

        evaled_points.append(x_to_evaluate)
        print evaled_points
        plt.scatter(np.array(evaled_points)[:, 0], np.array(evaled_points)[:, 1])
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.show()


if __name__ == '__main__':
    main()
