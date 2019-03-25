import numpy as np

from openravepy import DOFAffine
from planners.mcts_utils import make_action_hashable
from manipulation.primitives.savers import DynamicEnvironmentStateSaver
from trajectory_representation.operator import Operator

from problem_environment import ProblemEnvironment
from mover_problem import MoverProblem

## mover library utility functions
from mover_library.utils import set_robot_config, grab_obj, two_arm_pick_object, two_arm_place_object, \
    get_trajectory_length, visualize_path

OBJECT_ORIGINAL_COLOR = (0, 0, 0)
COLLIDING_OBJ_COLOR = (0, 1, 1)
TARGET_OBJ_COLOR = (1, 0, 0)


class MinimumDisplacementRemoval(ProblemEnvironment):
    def __init__(self, problem_idx):
        ProblemEnvironment.__init__(self)
        problem = MoverProblem(self.env, problem_idx)
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
        self.infeasible_reward = -2
        self.is_init_pick_node = True

        self.init_saver = DynamicEnvironmentStateSaver(self.env)
        self.problem_config['env'] = self.env
        self.objects_currently_not_in_goal = []
        self.swept_volume = None

    def set_objects_not_in_goal(self, objects_not_in_goal):
        self.objects_currently_not_in_goal = objects_not_in_goal

    def set_swept_volume(self, swept_volume):
        self.swept_volume = swept_volume

    def get_objs_in_region(self, region_name):
        movable_objs = self.objects
        objs_in_region = []
        for obj in movable_objs:
            if self.regions[region_name].contains(obj.ComputeAABB()):
                objs_in_region.append(obj)
        return objs_in_region

    def get_region_containing(self, obj):
        return self.regions['entire_region']

    def compute_place_reward(self, operator_instance):
        assert len(self.robot.GetGrabbed()) == 1
        prev_objects_not_in_goal = self.objects_currently_not_in_goal
        object_held = self.robot.GetGrabbed()[0]
        two_arm_place_object(object_held, self.robot, operator_instance.continuous_parameters)
        new_objects_not_in_goal = self.get_objs_in_collision(self.swept_volume, 'entire_region') # takes about 0.0284 seconds
        if len(prev_objects_not_in_goal) - len(new_objects_not_in_goal) > 0:
            distance_travelled = get_trajectory_length(operator_instance.low_level_motion)  # 0.3 ms
            reward = min(1.0 / distance_travelled, 2)
        else:
            distance_travelled = get_trajectory_length(operator_instance.low_level_motion)
            reward = max(-distance_travelled, self.infeasible_reward)
        return reward, new_objects_not_in_goal

    def apply_action_and_get_reward(self, operator_instance, is_op_feasible):
        if is_op_feasible != 'HasSolution':
            reward = self.infeasible_reward
        else:
            if operator_instance.type == 'two_arm_pick':
                two_arm_pick_object(operator_instance.discrete_parameters['object'],
                                    self.robot, operator_instance.continuous_parameters)
                reward = 0
            elif operator_instance.type == 'two_arm_place':
                reward, new_objects_not_in_goal = self.compute_place_reward(operator_instance)
                self.objects_currently_not_in_goal = new_objects_not_in_goal # todo I can potentially save time by keeping the reward in the node
            else:
                raise NotImplementedError

        return reward

    def reset_to_init_state(self, node):
        assert node.is_init_node, "None initial node passed to reset_to_init_state"
        saver = node.state_saver
        saver.Restore()
        self.curr_state = self.get_state()
        self.objects_currently_not_in_goal = node.objects_not_in_goal

        if not self.init_which_opreator != 'two_arm_pick':
            grab_obj(self.robot, self.curr_obj)

        if node.parent_action is not None:
            is_parent_action_pick = node.parent_action.type == 'two_arm_pick'
        else:
            is_parent_action_pick = False

        if is_parent_action_pick:
            two_arm_pick_object(node.parent_action.discrete_parameters['object'], self.robot,
                                node.parent_action.continuous_parameters)

        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    def check_reachability_precondition(self, operator_instance):
        motion_planning_region_name = 'entire_region'
        goal_robot_xytheta = operator_instance.continuous_parameters['base_pose']

        if operator_instance.low_level_motion is not None:
            motion = operator_instance.low_level_motion
            status = 'HasSolution'
            return motion, status

        motion, status = self.get_base_motion_plan(goal_robot_xytheta, motion_planning_region_name)
        return motion, status

    def is_goal_reached(self):
        return len(self.objects_currently_not_in_goal) == 0

    def which_operator(self):
        if self.is_pick_time():
            return 'two_arm_pick'
        else:
            return 'two_arm_place'

    @staticmethod
    def check_parameter_feasibility_precondition(operator_instance):
        if operator_instance.continuous_parameters['base_pose'] is None:
            return False
        else:
            return True

    def apply_operator_instance(self, operator_instance):
        if not self.check_parameter_feasibility_precondition(operator_instance):
            operator_instance.update_low_level_motion(None)
            return self.infeasible_reward

        motion_plan, status = self.check_reachability_precondition(operator_instance)
        operator_instance.update_low_level_motion(motion_plan)
        reward = self.apply_action_and_get_reward(operator_instance, status)

        return reward

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








