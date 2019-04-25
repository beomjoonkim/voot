from planning_problem_instantiator import PlanningProblemInstantiator
from problem_environments.conveyor_belt_env import ConveyorBelt


class ConveyorBeltInstantiator(PlanningProblemInstantiator):
    def __init__(self, problem_idx, domain_name, n_actions_per_node):
        PlanningProblemInstantiator.__init__(self, domain_name)
        self.environment = ConveyorBelt(problem_idx, n_actions_per_node)
        self.environment.set_objects_not_in_goal(self.environment.objects)





