from problem_instantiators.conveyor_belt_instantiator import ConveyorBeltInstantiator
from generators.feasibility_checkers.pick_feasibility_checker import PickFeasibilityChecker

from mover_library.utils import get_pick_domain
from generators.uniform import UniformGenerator
from trajectory_representation.operator import Operator
from mover_library import utils
import numpy as np
import random
import pickle

class FakeNode:
    def __init__(self, problem_env):
        operator_type = 'two_arm_pick'
        discrete_parameters = {'object': problem_env.objects[-1]}
        self.operator_skeleton = Operator(operator_type, discrete_parameters, None, None)


def main():
    random_seed = 0
    np.random.seed(random_seed)
    random.seed(random_seed)


    problem_idx = 3
    problem_instantiator = ConveyorBeltInstantiator(problem_idx, 'convbelt', 1000)
    environment = problem_instantiator.environment

    generator = UniformGenerator('two_arm_pick', environment)

    node = FakeNode(environment)
    action = generator.sample_next_point(node, 100)
    while action['is_feasible'] is not True:
        action = generator.sample_next_point(node, 100)

    init_goal_cam_transform = \
        np.array([[-0.99977334, 0.00150049, 0.02123698, -1.10846174],
                  [-0.01359773, 0.72254398, -0.69119122, 7.63502693],
                  [-0.01638178, -0.69132333, -0.72235981, 10.50135326],
                  [0., 0., 0., 1.]])
    environment.env.SetViewer('qtcoin')
    viewer = environment.env.GetViewer()
    viewer.SetCamera(init_goal_cam_transform)

    init_poses = pickle.load(open('./aaai_visualization_init_poses.pkl', 'r'))
    [utils.set_obj_xytheta(pose, obj) for obj, pose in zip(environment.objects, init_poses)]
    import pdb; pdb.set_trace()
    goal_poses = pickle.load(open('./aaai_visualization_goal_poses.pkl', 'r'))
    [utils.set_obj_xytheta(pose, obj) for obj,pose in zip(environment.objects, goal_poses)]
    import pdb; pdb.set_trace()
    utils.two_arm_pick_object(environment.objects[-1], environment.robot, action)
    utils.set_robot_config([-1.74301404,  0.11800224, np.pi], environment.robot)
    environment.env.Remove(environment.objects[0])
    import pdb; pdb.set_trace()




if __name__ == '__main__':
    main()
