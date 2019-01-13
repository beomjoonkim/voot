import sys
import numpy as np

sys.path.append('../mover_library/')
from generator import Generator
from planners.mcts_utils import make_action_executable

from utils import pick_parameter_distance, place_parameter_distance
from doo_utils.doo_tree import BinaryDOOTree
import matplotlib.pyplot as plt


class DOOGenerator(Generator):
    def __init__(self, operator_name, problem_env, explr_p):
        Generator.__init__(self, operator_name, problem_env)
        self.explr_p = explr_p
        self.doo_tree = BinaryDOOTree(self.domain)  # this depends on the problem
        self.update_flag = 'update_me'

    def sample_next_point(self, node, n_iter):
        self.update_evaled_values(node)
        self.doo_tree.update_evaled_values(self.evaled_actions, self.evaled_q_values)
        #print 'beginning', [l.f_value for l in self.doo_tree.leaves]

        for i in range(n_iter):
            action_parameters, doo_node = self.choose_next_point()
            action, status = self.feasibility_checker.check_feasibility(node, action_parameters)

            if status == 'HasSolution':
                self.doo_tree.expand_node(self.update_flag, doo_node)
                print "Found feasible sample"
                break
            else:
                self.evaled_actions.append(action_parameters)
                self.evaled_q_values.append(self.problem_env.infeasible_reward)
                #print 'infeasible', [l.f_value for l in self.doo_tree.leaves]
                self.doo_tree.expand_node(self.problem_env.infeasible_reward, doo_node)
                #print 'infeasible', [l.f_value for l in self.doo_tree.leaves]

        return action

    def choose_next_point(self):
        next_node = self.doo_tree.get_next_node_to_evaluate()
        x_to_evaluate = next_node.x_value
        return x_to_evaluate, next_node


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

    import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()
