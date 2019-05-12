import numpy as np
import time

from generator import Generator
from doo_utils.doo_tree import BinaryDOOTree
from mover_library.utils import pick_parameter_distance, place_parameter_distance

import matplotlib.pyplot as plt
import copy


class DOOGenerator(Generator):
    def __init__(self, operator_skeleton, problem_env, explr_p):
        Generator.__init__(self, operator_skeleton.type, problem_env)
        self.explr_p = explr_p
        self.x_min = copy.deepcopy(self.domain[0])
        self.x_max = copy.deepcopy(self.domain[1])
        self.domain[0] = self.normalize_x_value(self.domain[0])  # (self.domain[0] - self.x_min) / (self.x_max-self.x_min)
        self.domain[1] = self.normalize_x_value(self.domain[1])  # (self.domain[1] - self.x_min) / (self.x_max-self.x_min)
        self.idx_to_update = None

        operator_name = operator_skeleton.type
        if operator_name == 'two_arm_pick':
            target_object = operator_skeleton.discrete_parameters['object']
            if type(target_object) == str:
                target_object = self.problem_env.env.GetKinBody(target_object)
            dist_fn = lambda x, y: pick_parameter_distance(target_object, x, y)
        elif operator_name == 'two_arm_place':
            dist_fn = place_parameter_distance
        elif operator_name.find('_paps') != -1:
            n_actions = int(operator_name.split('_')[0])

            def dist_fn(x, y):
                x_obj_placements = np.split(x, n_actions)
                y_obj_placements = np.split(y, n_actions)
                dist = 0
                for x, y in zip(x_obj_placements, y_obj_placements):
                    dist += place_parameter_distance(x, y, 1)
                return dist
        elif operator_name.find('synthe') != -1:
            def dist_fn(x, y):
                return np.linalg.norm(x - y)
        else:
            print "Wrong operator name"
            raise ValueError
        self.doo_tree = BinaryDOOTree(self.domain, self.explr_p, dist_fn)
        self.update_flag = 'update_me'

    def sample_next_point(self, node, n_iter):
        self.update_evaled_values(node)
        #normalized_evaled_actions = [self.normalize_x_value(a) for a in self.evaled_actions]
        self.doo_tree.update_evaled_values(self.evaled_actions, self.evaled_q_values,
                                           self.problem_env.infeasible_reward,
                                           self.idx_to_update)
        print "DOO sampling..."
        stime = time.time()
        action, status, doo_node, action_parameters = self.sample_feasible_action(node, n_iter)
        print "Sampling time:", time.time()-stime

        if status == 'HasSolution':
            self.evaled_actions.append(action_parameters)
            self.doo_tree.expand_node(self.update_flag, doo_node)
            self.evaled_q_values.append(self.update_flag)
            self.idx_to_update = len(self.evaled_actions) - 1  # this assumes that we will use our algorithm, not UCT
            print "Found feasible sample"
        else:
            # I had this  bug where I was not updating when the action was infeasible. Now I've fixed it,
            # DOO should work better I think
            self.evaled_actions.append(action_parameters)
            self.doo_tree.expand_node(-2, doo_node)
            self.evaled_q_values.append(-2)

        return action

    def update_evaled_values(self, node):
        executed_actions_in_node = node.Q.keys()
        executed_action_values_in_node = node.Q.values()
        if len(executed_action_values_in_node) == 0:
            return

        if self.idx_to_update is not None:
            found = False
            for a, q in zip(executed_actions_in_node, executed_action_values_in_node):
                if np.all(np.isclose(self.evaled_actions[self.idx_to_update], a.continuous_parameters['action_parameters'])):
                    found = True
                    break
            try:
                assert found
            except AssertionError:
                print "idx to update not found"

            self.evaled_q_values[self.idx_to_update] = q

        assert np.array_equal(np.array(self.evaled_q_values).sort(), np.array(executed_action_values_in_node).sort()), \
            "Are you using N_r?"

    def sample_feasible_action(self, node, n_iter):
        for i in range(n_iter):
            action_parameters, doo_node = self.choose_next_point()
            action, status = self.feasibility_checker.check_feasibility(node, action_parameters)
            if status == 'HasSolution':
                break
            else:
                self.evaled_actions.append(action_parameters)
                self.doo_tree.expand_node(-2, doo_node)
                self.evaled_q_values.append(-2)
        print "Number of nodes in doo tree:", len(self.doo_tree.nodes)
        return action, status, doo_node, action_parameters

    def choose_next_point(self):
        next_node = self.doo_tree.get_next_point_and_node_to_evaluate()
        next_node.evaluated_x = next_node.cell_mid_point
        self.doo_tree.update_evaled_x_to_node(next_node.cell_mid_point, next_node)

        x_to_evaluate = next_node.cell_mid_point
        x_to_evaluate = self.unnormalize_x_value(x_to_evaluate)
        return x_to_evaluate, next_node

    def unnormalize_x_value(self, x_value):
        return x_value

    def normalize_x_value(self, x_value):
        return x_value


def test():
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
    test()
