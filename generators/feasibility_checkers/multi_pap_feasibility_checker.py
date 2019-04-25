from pick_feasibility_checker import PickFeasibilityChecker
from place_feasibility_checker import PlaceFeasibilityChecker
from mover_library.utils import release_obj, CustomStateSaver, two_arm_place_object
import numpy as np


class MultiPapFeasibilityChecker(PickFeasibilityChecker, PlaceFeasibilityChecker):
    def __init__(self, problem_env, n_paps):
        PlaceFeasibilityChecker.__init__(self, problem_env)
        self.n_paps = n_paps

    def check_feasibility(self, node, parameters):
        obj_placements = np.split(parameters, self.n_paps)
        cont_parameters = {'operator_name': 'two_paps', 'base_poses': [], 'object_poses': [],
                           'action_parameters': parameters, 'is_feasible': False}

        objs = node.operator_skeleton.discrete_parameters['objects']
        state_saver = CustomStateSaver(self.env)

        status = "NoSolution"
        for target_object, obj_placement in zip(objs, obj_placements):
            node.operator_skeleton.discrete_parameters['object'] = target_object

            self.problem_env.pick_object(target_object)
            place_cont_params, status = PlaceFeasibilityChecker.check_feasibility(self, node, obj_placement)

            if status == 'NoSolution':
                break
            else:
                cont_parameters['base_poses'].append(place_cont_params['base_pose'])
                cont_parameters['object_poses'].append(place_cont_params['object_pose'])
                two_arm_place_object(target_object, self.robot, {'base_pose': place_cont_params['base_pose']})
        state_saver.Restore()

        if status == "HasSolution":
            cont_parameters['is_feasible'] = True

        return cont_parameters, status

