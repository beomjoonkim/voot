from generators.generator import Generator
import numpy as np
import time


class DDPGGenerator(Generator):
    def __init__(self, op_name, problem_env, policy):
        Generator.__init__(self, op_name, problem_env)
        self.policy = policy
        self.epsilon = 0.95  # epsilon greedy exploration

    def sample_next_point(self, node, state, n_iter):
        #p_use_unif = np.power(self.epsilon, self.policy.epoch)
        p_use_unif = np.power(self.epsilon, self.policy.n_feasible_trajs)
        print p_use_unif
        use_unif = p_use_unif >= np.random.uniform()

        if not use_unif:
            print "Executing policy"
        for i in range(n_iter):
            if use_unif:
                action_parameters = self.sample_from_uniform()
            else:
                action_parameters = self.policy.predict(state).squeeze()

            action, status = self.feasibility_checker.check_feasibility(node, action_parameters)
            if status == 'HasSolution':
                # print "Found feasible sample"
                break
        print "Done"
        return action
