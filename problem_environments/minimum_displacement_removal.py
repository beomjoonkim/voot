import numpy as np

from openravepy import DOFAffine
from planners.mcts_utils import make_action_hashable
from manipulation.primitives.savers import DynamicEnvironmentStateSaver
from trajectory_representation.operator import Operator

from problem_environment import ProblemEnvironment
from minimum_displacement_removal_problem import MinimumDisplacementRemovalProblem

## mover library utility functions
from mover_library.utils import set_robot_config, grab_obj, two_arm_pick_object, two_arm_place_object, \
    get_trajectory_length, visualize_path

OBJECT_ORIGINAL_COLOR = (0, 0, 0)
COLLIDING_OBJ_COLOR = (0, 1, 1)
TARGET_OBJ_COLOR = (1, 0, 0)


class MinimumDisplacementRemoval(ProblemEnvironment):
    def __init__(self, problem_idx):
        ProblemEnvironment.__init__(self)
        problem = MinimumDisplacementRemovalProblem(self.env, problem_idx)
        self.problem_config = problem.get_problem_config()
        self.robot = self.env.GetRobots()[0]
        self.objects = self.problem_config['objects']
        self.regions = {'entire_region': self.problem_config['entire_region']}
        self.init_base_conf = self.problem_config['init_base_config']
        self.goal_base_conf = self.problem_config['goal_base_config']
        self.problem_idx = self.problem_config['problem_idx']

        self.init_saver = DynamicEnvironmentStateSaver(self.env)
        self.robot = self.env.GetRobots()[0]
        self.objects = self.problem_config['objects']
        self.regions = {'entire_region': self.problem_config['entire_region']}

        self.is_init_pick_node = True

        self.init_saver = DynamicEnvironmentStateSaver(self.env)
        self.problem_config['env'] = self.env
        self.swept_volume = None

    def set_swept_volume(self, swept_volume):
        self.swept_volume = swept_volume

    def get_region_containing(self, obj):
        return self.regions['entire_region']

    def compute_place_reward(self, operator_instance):
        # todo I can potentially save time by keeping the reward in the node
        assert len(self.robot.GetGrabbed()) == 1
        prev_objects_not_in_goal = self.objects_currently_not_in_goal
        object_held = self.robot.GetGrabbed()[0]
        two_arm_place_object(object_held, self.robot, operator_instance.continuous_parameters)
        new_objects_not_in_goal = self.get_objs_in_collision(self.swept_volume,
                                                             'entire_region')  # takes about 0.0284 seconds
        if len(prev_objects_not_in_goal) - len(new_objects_not_in_goal) > 0:
            distance_travelled = get_trajectory_length(operator_instance.low_level_motion)  # 0.3 ms
            reward = min(1.0 / distance_travelled, 2)
        else:
            distance_travelled = get_trajectory_length(operator_instance.low_level_motion)
            reward = max(-distance_travelled, self.infeasible_reward)
        return reward, new_objects_not_in_goal

    def is_goal_reached(self):
        goal_achieved = len(self.objects_currently_not_in_goal) == 0
        return goal_achieved

    def get_applicable_op_skeleton(self):
        op_name = self.which_operator()
        if op_name == 'two_arm_place':
            op = Operator(operator_type=op_name,
                          discrete_parameters={'region': self.regions['entire_region']},
                          continuous_parameters=None,
                          low_level_motion=None)
        else:
            op = Operator(operator_type=op_name,
                          discrete_parameters={'object': self.objects_currently_not_in_goal[0]},
                          continuous_parameters=None,
                          low_level_motion=None)
        return op








