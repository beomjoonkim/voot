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

    def is_ucb_step(self):
        n_arms = len(self.arms)
        if n_arms < 10:
            return False
        else:
            if self.n_ucb_iterations < self.widening_parameter:
                print "UCB iteration"
                self.n_ucb_iterations += 1
                return True
            else:
                self.n_ucb_iterations = 0
                return False

    def ucb_upperbound(self, arm):
        ucb_value = arm.expected_value + self.ucb_parameter * np.sqrt(np.log(self.n_evaluations) / float(arm.n_visited))
        return ucb_value

    def perform_ucb(self):
        best_value = -np.inf
        best_arm = self.arms[0]
        for arm in self.arms:
            ucb_value = self.ucb_upperbound(arm)
            if ucb_value > best_value:
                best_arm = arm
                best_value = ucb_value

        import pdb;pdb.set_trace()

        return best_arm

    def sample_next_point(self):
        dim_parameters = self.domain.shape[-1]
        domain_min = self.domain[0]
        domain_max = self.domain[1]

        x = np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()
        return x

    def choose_next_point(self, dummy, dummy2):
        if self.is_ucb_step():
            print "Evaluating existing point using UCB"
            x = self.perform_ucb()
        else:
            print "Evaluating a new point"
            x = self.sample_next_point()
            x = BanditArm(x)

        return x

    def update_evaluated_arms(self, evaluated_arm, new_reward):
        self.n_evaluations += 1
        evaluated_arm.update_value(new_reward)
        import pdb;pdb.set_trace()
        if not(evaluated_arm in self.arms):
            self.arms.append(evaluated_arm)

