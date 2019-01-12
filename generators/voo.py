import sys
import numpy as np
sys.path.append('../mover_library/')
from samplers import gaussian_randomly_place_in_region
from generator import Generator
from utils import pick_parameter_distance, place_parameter_distance
from planners.mcts_utils import make_action_executable


class VOOGenerator(Generator):
    def __init__(self, operator_name, problem_env, explr_p, c1):
        Generator.__init__(self, operator_name, problem_env)
        self.explr_p = explr_p
        self.evaled_actions = []
        self.evaled_q_values = []
        self.c1 = c1

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

        rnd = np.random.random() # this should lie outside
        is_sample_from_best_v_region = rnd < 1 - self.explr_p and len(self.evaled_actions) > 1 and \
                                       np.max(self.evaled_q_values) > self.problem_env.infeasible_reward
        print "VOO sampling..."
        for i in range(n_iter):
            if is_sample_from_best_v_region:
                action_parameters = self.sample_from_best_voronoi_region(node)
            else:
                action_parameters = self.sample_from_uniform()
            action, status = self.feasibility_checker.check_feasibility(node,  action_parameters)

            if status == 'HasSolution':
                print "Found feasible sample"
                break
            else:
                # this does not add any information
                #self.evaled_actions.append(action_parameters)
                #self.evaled_q_values.append(self.problem_env.infeasible_reward)
                pass

        return action

    def sample_from_best_voronoi_region(self, node):
        operator = node.operator
        obj = node.obj
        region = node.region
        if operator == 'two_arm_pick':
            params = self.sample_pick_from_best_voroi_region(obj)
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
            #print "Gaussian place sampling, counter", counter, len(other_dists)
            variance = 0.5*(self.domain[1] - self.domain[0]) / counter
            new_parameters = np.random.normal(best_evaled_action, variance)
            new_parameters = np.clip(new_parameters, self.domain[0], self.domain[1])

            best_dist = place_parameter_distance(new_parameters, best_evaled_action, self.c1)
            other_dists = np.array([place_parameter_distance(other, new_parameters, self.c1) for other in other_actions])
            counter += 1
        return new_parameters

    def sample_pick_from_best_voroi_region(self, obj):
        best_dist = np.inf
        other_dists = np.array([-1])
        counter = 1

        best_action_idxs = np.argwhere(self.evaled_q_values == np.amax(self.evaled_q_values))
        best_action_idxs = best_action_idxs.reshape((len(best_action_idxs, )))
        best_action_idx = np.random.choice(best_action_idxs)
        best_evaled_action = self.evaled_actions[best_action_idx]
        other_actions = self.evaled_actions

        while np.any(best_dist > other_dists):
            variance = 0.5*(self.domain[1] - self.domain[0]) / counter
            new_parameters = np.random.normal(best_evaled_action, variance)
            new_parameters = np.clip(new_parameters, self.domain[0], self.domain[1])

            best_dist = pick_parameter_distance(obj, new_parameters, best_evaled_action)
            other_dists = np.array([pick_parameter_distance(obj, other, new_parameters) for other in
                                    other_actions])
            counter += 1
            #print "Gaussian pick sampling, variance and counter", variance, counter, len(other_dists)

        return new_parameters



