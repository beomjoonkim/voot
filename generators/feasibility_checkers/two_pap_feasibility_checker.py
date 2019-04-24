from pick_feasibility_checker import PickFeasibilityChecker
from place_feasibility_checker import PlaceFeasibilityChecker
from mover_library.utils import two_arm_pick_object, release_obj, CustomStateSaver, grab_obj, set_robot_config
import pickle
import re


class TwoPapFeasibilityChecker(PickFeasibilityChecker, PlaceFeasibilityChecker):
    def __init__(self, problem_env):
        PlaceFeasibilityChecker.__init__(self, problem_env)

    def check_feasibility(self, node, parameters):
        obj1_place = parameters[0:3]
        obj2_place = parameters[3:]

        obj_placements = [obj1_place, obj2_place]
        cont_parameters = {'operator_name': 'two_paps', 'base_poses': [],
                           'object_poses': [], 'action_parameters': parameters, 'is_feasible':False}

        objs = node.operator_skeleton.discrete_parameters['objects']

        state_saver = CustomStateSaver(self.env)

        status = "NoSolution"
        for target_object, obj_placement in zip(objs, obj_placements):
            node.operator_skeleton.discrete_parameters['object'] = target_object

            self.problem_env.pick_object(target_object)
            place_cont_params, status = PlaceFeasibilityChecker.check_feasibility(self, node, obj_placement)
            release_obj(self.robot, target_object)

            if status == 'NoSolution':
                break
            else:
                cont_parameters['base_poses'].append(place_cont_params['base_pose'])
                cont_parameters['object_poses'].append(place_cont_params['object_pose'])

        state_saver.Restore()

        if status == "HasSolution":
            cont_parameters['is_feasible'] = True
        return cont_parameters, status

