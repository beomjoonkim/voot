import sys
import numpy as np
sys.path.append('../mover_library/')
from samplers import gaussian_randomly_place_in_region
from generator import Generator
from utils import pick_parameter_distance, place_parameter_distance
from planners.mcts_utils import make_action_executable


class VOOGenerator(Generator):
    def __init__(self, operator_name, problem_env, explr_p):
        Generator.__init__(self, operator_name, problem_env)
        self.explr_p = explr_p
        self.evaled_actions = []
        self.evaled_q_values = []

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
            rnd = np.random.random()
            if rnd < 1 - self.explr_p and len(self.evaled_actions) > 0 \
                    and np.max(self.evaled_q_values) > self.problem_env.infeasible_reward:
                action_parameters = self.sample_from_best_voronoi_region(node)
            else:
                action_parameters = self.sample_from_uniform()
            action, status = self.feasibility_checker.check_feasibility(node,  action_parameters)

            if status == 'HasSolution':
                print "Found feasible sample"
                break
            #else:
            #    self.evaled_actions.append(action_parameters)
            #    self.evaled_q_values.append(self.problem_env.infeasible_reward)

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

        best_action_idxs = np.argwhere(self.evaled_q_values == np.amax(self.evaled_q_values))
        best_action_idxs = best_action_idxs.reshape((len(best_action_idxs,)))
        best_action_idx = np.random.choice(best_action_idxs)
        best_evaled_action = self.evaled_actions[best_action_idx]
        other_actions = self.evaled_actions

        while np.any(best_dist > other_dists):
            #print "Gaussian place sampling, counter", counter
            variance = 0.5*(self.domain[1] - self.domain[0]) / counter
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
            #print "Gaussian pick sampling, counter", counter
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



