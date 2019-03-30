import cPickle as pickle

from planning_problem_instantiator import PlanningProblemInstantiator
from problem_environments.conveyor_belt_env import ConveyorBelt


class ConveyorBeltInstantiator(PlanningProblemInstantiator):
    def __init__(self, domain_name):
        PlanningProblemInstantiator.__init__(self, domain_name)
        self.environment = ConveyorBelt(problem_idx=1)
        self.environment.set_objects_not_in_goal(self.environment.objects)

    def load_swept_volume(self):
        swept_volume_file_name = './problem_environments/mover_domain_problems/fetching_path_' + \
                                 str(self.environment.problem_idx) + '.pkl'

        return pickle.load(open(swept_volume_file_name, 'r'))



