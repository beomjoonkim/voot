from planning_problem_instantiator import PlanningProblemInstantiator
from problem_environments.conveyor_belt_env import ConveyorBelt


class ConveyorBeltInstantiator(PlanningProblemInstantiator):
    def __init__(self, problem_idx, domain_name):
        PlanningProblemInstantiator.__init__(self, domain_name)
        self.environment = ConveyorBelt(problem_idx)
        self.environment.set_objects_not_in_goal(self.environment.objects)





