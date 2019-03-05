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


class StoVOO(VOO):
    def __init__(self, domain, ucb_parameter, widening_parameter, explr_p, distance_fn = None):
        VOO.__init__(self, domain, explr_p, distance_fn)
        self.n_ucb_iterations = 0
        self.arms = []
        self.n_evaluations = 0
        self.widening_parameter = widening_parameter
        self.ucb_parameter = ucb_parameter

    def is_ucb_step(self):
        n_arms = len(self.arms)
        # todo I think this parameter should increase with the number of arms;
        # todo I also think you should begin with a set of values

        """
        progressive_widening_value = self.widening_parameter * self.n_evaluations
        print n_arms, progressive_widening_value
        if n_arms > progressive_widening_value:
            return True
        else:
            return False
        """

        if n_arms < 10:
            return False
        else:
            if self.n_ucb_iterations < self.widening_parameter / float(self.n_evaluations):
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

        return best_arm

    def choose_next_point(self, dummy, dummy2):
        if self.is_ucb_step():
            print "Evaluating existing point using UCB"
            x = self.perform_ucb()
        else:
            print "Evaluating a new point"
            evaled_x = [a.x_value for a in self.arms]
            evaled_y = [a.expected_value for a in self.arms]
            x = self.sample_next_point(evaled_x, evaled_y)
            x = BanditArm(x)

        self.n_evaluations += 1
        return x

    def update_evaluated_arms(self, evaluated_arm, new_reward):
        evaluated_arm.update_value(new_reward)
        if not(evaluated_arm in self.arms):
            self.arms.append(evaluated_arm)

