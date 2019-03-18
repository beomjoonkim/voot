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
    def __init__(self, domain, ucb_parameter, widening_parameter, explr_p, distance_fn = None,
                 is_progressive_widening=True):
        VOO.__init__(self, domain, explr_p, distance_fn)
        self.n_ucb_iterations = 0
        self.arms = []
        self.n_evaluations = 0
        self.widening_parameter = widening_parameter
        self.ucb_parameter = ucb_parameter
        self.is_progressive_widening = is_progressive_widening

    def is_ucb_step(self):
        n_arms = len(self.arms)
        if self.is_progressive_widening:
            if n_arms <= self.widening_parameter * self.n_evaluations:
                # self.widening_parameter = 0.1
                # n_evaluations = 0,1,2,3,4,5,6,7,8,9,10,11,...,20
                # w*n = 0, 0.1, 0.2, 0.3,..., 0.9, 1, 2
                # n_arms = 0, w*n=0 -> new arm
                # n_arms = 1, w*n=0.1 -> ucb
                # n_arms = 1, w*n=0.2 -> ucb
                # n_arms = 1, w*n=0.3 -> ucb
                #   ...
                # n_arms = 1, w*n=0.9 -> ucb
                return False
            else:
                return True
        else:
            if n_arms < 10:
                return False
            if self.n_ucb_iterations < self.widening_parameter: #/ float(self.n_evaluations):
                print "UCB iteration"
                self.n_ucb_iterations += 1
                return True
            else:
                print "VOO iteration"
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

    def choose_next_point(self):
        if self.is_ucb_step():
            print "Evaluating existing point using UCB"
            x = self.perform_ucb()
        else:
            print "Evaluating a new point"
            evaled_x = [a.x_value for a in self.arms]
            evaled_y = [a.expected_value for a in self.arms]
            x = self.sample_next_point(evaled_x, evaled_y)
            x = BanditArm(x)
        return x

    def update_evaluated_arms(self, evaluated_arm, new_reward):
        evaluated_arm.update_value(new_reward)
        self.n_evaluations += 1
        if not(evaluated_arm in self.arms):
            self.arms.append(evaluated_arm)

