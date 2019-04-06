import cPickle as pickle

from planning_problem_instantiator import PlanningProblemInstantiator
from problem_environments.minimum_displacement_removal import MinimumDisplacementRemoval
from mover_library.utils import two_arm_pick_object


class MinimumConstraintRemovalInstantiator(PlanningProblemInstantiator):
    def __init__(self, domain_name):
        PlanningProblemInstantiator.__init__(self, domain_name)

        self.environment = MinimumDisplacementRemoval(problem_idx=1)
        swept_volume_to_clear_obstacles_from = self.load_swept_volume()
        initial_collisions = self.environment.get_objs_in_collision(swept_volume_to_clear_obstacles_from, 'entire_region')
        """
        for o in initial_collisions[1:]:
            self.environment.env.Remove(o)
            self.environment.objects.remove(o)
            initial_collisions.remove(o)
        """
        first_pick = pickle.load(open('tmp.pkl', 'r'))
        obj = self.environment.env.GetKinBody(first_pick.discrete_parameters['object'])
        two_arm_pick_object(obj, self.environment.robot, first_pick.continuous_parameters)

        self.environment.set_objects_not_in_goal(initial_collisions)
        self.environment.set_swept_volume(swept_volume_to_clear_obstacles_from)

    def load_swept_volume(self):
        swept_volume_file_name = './problem_environments/mover_domain_problems/fetching_path_' + \
                                 str(self.environment.problem_idx) + '.pkl'

        return pickle.load(open(swept_volume_file_name, 'r'))



