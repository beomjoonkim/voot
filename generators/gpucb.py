import sys
import numpy as np
sys.path.append('../mover_library/')
from generator import Generator
from utils import pick_parameter_distance, place_parameter_distance
from planners.mcts_utils import make_action_executable

from gpucb_utils.gp import StandardContinuousGP
from gpucb_utils.functions import UCB, Domain
from gpucb_utils.bo import BO


class GPUCBGenerator(Generator):
    def __init__(self, operator_name, problem_env, explr_p):
        Generator.__init__(self, operator_name, problem_env)
        self.explr_p = explr_p
        self.evaled_actions = []
        self.evaled_q_values = []

        self.gp = StandardContinuousGP(3)
        self.acq_fcn = UCB(zeta=explr_p, gp=self.gp)
        self.domain = Domain(0, self.domain)
        self.gp_optimizer = BO(self.gp, self.acq_fcn, self.domain)  # this depends on the problem

    def update_evaled_values(self, node):
        executed_actions_in_node = node.Q.keys()
        executed_action_values_in_node = node.Q.values()

        for action, q_value in zip(executed_actions_in_node, executed_action_values_in_node):
            executable_action = make_action_executable(action)
            is_in_array = [np.array_equal(executable_action['action_parameters'], a) for a in self.evaled_actions]
            is_action_included = np.any(is_in_array)

            if not is_action_included:
                self.evaled_actions.append(executable_action['action_parameters'])
                self.evaled_q_values.append(q_value)
            else:
                # update the value if the action is included
                self.evaled_q_values[np.where(is_in_array)[0][0]] = q_value

    def sample_next_point(self, node, n_iter):
        self.update_evaled_values(node)

        for i in range(n_iter):
            action_parameters = self.gp_optimizer.choose_next_point(self.evaled_actions, self.evaled_q_values)
            if i > 300:
                # make it escape the local optima
                action_parameters = self.sample_from_uniform()
            action, status = self.feasibility_checker.check_feasibility(node,  action_parameters)

            if status == 'HasSolution':
                print "Found feasible sample"
                break
            else:
                #takes too long
                self.evaled_actions.append(action_parameters)
                self.evaled_q_values.append(self.problem_env.infeasible_reward)
                pass

        return action

    def sample_from_best_voronoi_region(self, node):
        # todo write below
        operator = node.operator
        obj = node.obj
        region = node.region
        if operator == 'two_arm_pick':
            params = self.sample_pick_from_best_voroi_region()
        elif operator == 'two_arm_place':
            params = self.sample_place_from_best_voroi_region()
        return params

    def sample_place_from_best_voroi_region(self):
        best_dist = np.inf
        other_dists = np.array([-1])
        counter = 1

        best_action_idxs = np.argwhere(self.evaled_q_values == np.amax(self.evaled_q_values)).squeeze()
        best_action_idx = np.random.choice(best_action_idxs)
        best_evaled_action = self.evaled_actions[best_action_idx]
        other_actions = self.evaled_actions

        while np.any(best_dist > other_dists):
            print "Gaussian place sampling, counter", counter
            variance = (self.domain[1] - self.domain[0]) / counter
            new_parameters = np.random.normal(best_evaled_action, variance)

            new_parameters = np.clip(new_parameters, self.domain[0], self.domain[1])
            best_dist = place_parameter_distance(new_parameters, best_evaled_action)
            other_dists = np.array([place_parameter_distance(other, new_parameters) for other in other_actions])
            counter += 1
        return new_parameters

    def sample_pick_from_best_voroi_region(self):
        best_dist = np.inf
        other_dists = np.array([-1])
        counter = 1

        best_evaled_action = self.evaled_actions[np.argmax(self.evaled_q_values)]
        other_actions = self.evaled_actions

        '''
        grasp_params = pick_parameters[0:3]
        portion = pick_parameters[3]
        base_angle = pick_parameters[4]
        facing_angle = pick_parameters[5]
        '''

        while np.any(best_dist > other_dists):
            print "Gaussian pick sampling, counter", counter
            best_ir_parameters = best_evaled_action[3:]

            var_ir = np.array([0.3, 30*np.pi/180., 10*np.pi/180]) / float(counter)
            ir_parameters = np.random.normal(best_ir_parameters, var_ir)

            best_action_grasp_params = best_evaled_action[0:3]
            var_grasp = np.array([0.5, 0.2, 0.2]) / float(counter)
            grasp_params = np.random.normal(best_action_grasp_params, var_grasp)

            new_parameters = np.hstack([grasp_params, ir_parameters])
            new_parameters = np.clip(new_parameters, self.domain[0], self.domain[1])
            best_dist = pick_parameter_distance(new_parameters, best_evaled_action, self.domain)
            other_dists = np.array([pick_parameter_distance(other, new_parameters, self.domain) for other in
                                    other_actions])
            counter += 1

        return new_parameters



