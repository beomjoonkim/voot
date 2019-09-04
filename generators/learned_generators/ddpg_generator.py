from generators.generator import Generator
import numpy as np
import time


class DDPGGenerator(Generator):
    def __init__(self, op_name, problem_env, policy):
        Generator.__init__(self, op_name, problem_env)
        self.policy = policy
        self.epsilon = 0.99 # epsilon greedy exploration

    def sample_next_point(self, node, state, n_iter):
        p_use_unif = np.power(self.epsilon, self.policy.n_weight_updates)
        use_unif = p_use_unif >= np.random.uniform()

        for i in range(n_iter):
            if use_unif:
                action_parameters = self.sample_from_uniform()
            else:
                print "Executing policy"
                action_parameters = self.policy.predict(state).squeeze()

            action, status = self.feasibility_checker.check_feasibility(node, action_parameters)
            if status == 'HasSolution':
                # print "Found feasible sample"
                break
        return action
