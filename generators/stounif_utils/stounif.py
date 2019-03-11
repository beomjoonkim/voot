from generators.voo_utils.voo import VOO
import numpy as np


class BanditArm:
    def __init__(self, x_value):
        self.x_value = x_value
        self.sum_rewards = 0
        self.expected_value = 0
        self.n_visited = 0

    def update_value(self, reward):
        self.n_visited += 1
        self.sum_rewards += reward
        self.expected_value = self.sum_rewards / float(self.n_visited)


class StoUniform:
    def __init__(self, domain, ucb_parameter, widening_parameter):
        self.domain = domain
        self.ucb_parameter = ucb_parameter
        self.widening_parameter = widening_parameter

        self.n_ucb_iterations = 0
        self.arms = []
        self.n_evaluations = 0
        self.widening_parameter = widening_parameter
        self.ucb_parameter = ucb_parameter

    def sample_next_point(self):
        dim_parameters = self.domain.shape[-1]
        domain_min = self.domain[0]
        domain_max = self.domain[1]

        x = np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()
        return x

    def choose_next_point(self, dummy, dummy2):
        print "Evaluating a new point"
        x = self.sample_next_point()
        x = BanditArm(x)

        return x

    def update_evaluated_arms(self, evaluated_arm, new_reward):
        self.n_evaluations += 1
        evaluated_arm.update_value(new_reward)
        if not(evaluated_arm in self.arms):
            self.arms.append(evaluated_arm)

