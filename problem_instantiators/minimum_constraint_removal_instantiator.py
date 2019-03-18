import cPickle as pickle

from planning_problem_instantiator import PlanningProblemInstantiator
from problem_environments.minimum_displacement_removal import MinimumDisplacementRemoval


class MinimumConstraintRemovalInstantiator(PlanningProblemInstantiator):
    def __init__(self, domain_name):
        PlanningProblemInstantiator.__init__(self, domain_name)

        self.environment = MinimumDisplacementRemoval(problem_idx=1)
        swept_volume_to_clear_obstacles_from = self.load_swept_volume()
        initial_collisions = self.environment.get_objs_in_collision(swept_volume_to_clear_obstacles_from,
                                                                    'entire_region')
        self.environment.set_objects_not_in_goal(initial_collisions)

    def load_swept_volume(self):
        swept_volume_file_name = './problem_environments/mover_domain_problems/fetching_path_' + \
                                 str(self.environment.problem_idx) + '.pkl'

        return pickle.load(open(swept_volume_file_name, 'r'))



