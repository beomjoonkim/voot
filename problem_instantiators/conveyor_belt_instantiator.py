import cPickle as pickle

from planning_problem_instantiator import PlanningProblemInstantiator
from problem_environments.conveyor_belt_env import ConveyorBelt


from mover_library.utils import two_arm_pick_object, set_robot_config
class ConveyorBeltInstantiator(PlanningProblemInstantiator):
    def __init__(self, domain_name):
        PlanningProblemInstantiator.__init__(self, domain_name)
        self.environment = ConveyorBelt(problem_idx=1)
        self.environment.set_objects_not_in_goal(self.environment.objects)
        #  todo pickup the object
        #pick = pickle.load(open('tmp.pkl','r'))
        #two_arm_pick_object(self.environment.env.GetKinBody(pick.discrete_parameters['object']),  self.environment.robot, pick.continuous_parameters,)
        #set_robot_config(self.environment.init_base_conf, self.environment.robot)





