import numpy as np
import sys
import copy
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../mover_library/')
from conveyor_belt_problem import create_conveyor_belt_problem
from problem_environment import ProblemEnvironment
from trajectory_representation.operator import Operator
import cPickle as pickle

from mover_library.utils import *
from operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp
from manipulation.primitives.savers import DynamicEnvironmentStateSaver
from openravepy import DOFAffine, Environment
from manipulation.bodies.bodies import set_color

class ConveyorBelt(ProblemEnvironment):
    def __init__(self, problem_idx):
        self.problem_idx = problem_idx
        ProblemEnvironment.__init__(self, problem_idx)
        obj_setup = None
        self.problem_config = create_conveyor_belt_problem(self.env, obj_setup, problem_idx)
        self.objects = self.problem_config['objects']
        self.objects[0], self.objects[2] = self.objects[2] ,self.objects[0]
        set_color( self.objects[0] , [1,0,0] )

        self.init_base_conf = np.array([0, 1.05, 0])
        self.fetch_planner = None

        self.regions = {'entire_region': self.problem_config['entire_region'],
                        'object_region': self.problem_config['loading_region']}

        self.robot = self.problem_config['env'].GetRobots()[0]
        self.infeasible_reward = -2

        self.curr_state = self.get_state()

        self.init_saver = DynamicEnvironmentStateSaver(self.env)
        self.is_init_pick_node = True
        self.init_operator = 'two_arm_place'
        self.name = 'convbelt'

    def check_reachability_precondition(self, operator_instance):
        # we can potentially make this faster by just checking the collision at the door
        # actually, check collisions at 360 degree, with 20 degrees increments.
        # We don't need to call motion planner.

        # the condition is where the object is facing.
        grabbed_obj = self.robot.GetGrabbed()[0]
        obj_name = grabbed_obj.GetName()
        if obj_name.find('tobj') != -1:
            obj_theta = get_body_xytheta(grabbed_obj)[0][-1] * 180/np.pi
            if obj_theta < 0:
                obj_theta += 360
            assert obj_theta > 0
            no_solution = False
            if obj_name == 'tobj1' or obj_name == 'tobj4':
                if obj_theta < 45 or 135 < obj_theta < 225 or obj_theta > 315:
                    no_solution = True
            elif obj_name == 'tobj3':
                if (45 < obj_theta < 135) or (225 < obj_theta < 315):
                    no_solution = True
            if no_solution:
                return None, "NoSolution"

        goal_robot_xytheta = operator_instance.continuous_parameters['base_pose']

        if operator_instance.low_level_motion is not None:
            motion = operator_instance.low_level_motion
            status = 'HasSolution'
            return motion, status

        motion, status = self.get_base_motion_plan(goal_robot_xytheta, None)
        return motion, status

    def reset_to_init_state(self, node):
        assert node.is_init_node, "None initial node passed to reset_to_init_state"
        saver = node.state_saver
        saver.Restore()
        self.curr_state = self.get_state()
        self.objects_currently_not_in_goal = node.objects_not_in_goal

        if node.parent_action is not None:
            is_parent_action_pick = node.parent_action.type == 'two_arm_pick'
        else:
            is_parent_action_pick = False

        if is_parent_action_pick:
            obj = node.parent_action.discrete_parameters['object']
            grab_obj(self.robot, obj)

        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    def apply_action_and_get_reward(self, operator_instance, is_op_feasible, node):
        if is_op_feasible != 'HasSolution':
            reward = self.infeasible_reward
        else:
            if operator_instance.type == 'two_arm_pick':
                two_arm_pick_object(operator_instance.discrete_parameters['object'],
                                    self.robot, operator_instance.continuous_parameters)
                set_robot_config(self.init_base_conf, self.robot)
                reward = 1
            elif operator_instance.type == 'two_arm_place':
                reward, new_objects_not_in_goal = self.compute_place_reward(operator_instance)
                self.set_objects_not_in_goal(new_objects_not_in_goal)
            else:
                raise NotImplementedError

        return reward

    def apply_operator_instance(self, operator_instance, node):
        if not self.check_parameter_feasibility_precondition(operator_instance):
            operator_instance.update_low_level_motion(None)
            return self.infeasible_reward

        if operator_instance.type == 'two_arm_place':
            motion_plan, status = self.check_reachability_precondition(operator_instance)
            operator_instance.update_low_level_motion(motion_plan)
        else:
            status = "HasSolution"

        reward = self.apply_action_and_get_reward(operator_instance, status, node)
        return reward

    def compute_place_reward(self, operator_instance):
        assert len(self.robot.GetGrabbed()) == 1
        object_held = self.robot.GetGrabbed()[0]
        two_arm_place_object(object_held, self.robot, operator_instance.continuous_parameters)
        new_objects_not_in_goal = self.objects_currently_not_in_goal[1:]
        reward = self.objects.index(object_held)+1  # reward gradually increases
        return reward, new_objects_not_in_goal

    def is_goal_reached(self):
        return len(self.get_objs_in_region('object_region')) == len(self.objects)

    def load_object_setup(self):
        object_setup_file_name = './problem_environments/conveyor_belt_domain_problems/' + str(self.problem_idx) + '.pkl'
        if os.path.isfile(object_setup_file_name):
            obj_setup = pickle.load(open('./problem_environments/conveyor_belt_domain_problems/' + str(self.problem_idx) + '.pkl', 'r'))
            return obj_setup
        else:
            return None

    def save_object_setup(self):
        object_configs = {'object_poses': self.problem_config['obj_poses'],
                          'object_shapes': self.problem_config['obj_shapes'],
                          'obst_poses': self.problem_config['obst_poses'],
                          'obst_shapes': self.problem_config['obst_shapes']}
        pickle.dump(object_configs, open('./problem_environments/conveyor_belt_domain_problems/' + str(self.problem_idx) + '.pkl', 'wb'))

    def get_applicable_op_skeleton(self):
        op_name = self.which_operator()
        if op_name == 'two_arm_place':
            op = Operator(operator_type=op_name,
                          discrete_parameters={'region': self.regions['object_region']},
                          continuous_parameters=None,
                          low_level_motion=None)
        else:
            op = Operator(operator_type=op_name,
                          discrete_parameters={'object': self.objects_currently_not_in_goal[0]},
                          continuous_parameters=None,
                          low_level_motion=None)

        return op



