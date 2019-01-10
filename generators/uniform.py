import sys
import numpy as np
sys.path.append('../mover_library/')
from samplers import *
from generator import Generator


class UniformGenerator(Generator):
    def __init__(self, operator_name, problem_env):
        Generator.__init__(self, operator_name, problem_env)

    def sample_next_point(self, node, n_iter):
        dim_parameters = self.domain.shape[-1]
        domain_min = self.domain[0]
        domain_max = self.domain[1]
        for i in range(n_iter):
            action_parameters = np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()
            action, status = self.feasibility_checker.check_feasibility(node,  action_parameters)
            if status == 'HasSolution':
                print "Found feasible sample"
                break
        return action

