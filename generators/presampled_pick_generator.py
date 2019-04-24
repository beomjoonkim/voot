import re
from generators.generator import Generator
import pickle


class PreSampledPickGenerator(Generator):
    def __init__(self):
        self.picks = pickle.load(open('problem_environments/convbelt_picks.pkl', 'r'))

    def sample_next_point(self, node, n_iter):
        obj = node.operator_skeleton.discrete_parameters['object']
        obj_number = int(re.search(r'\d+', obj.GetName()).group())
        return self.picks[obj_number]

