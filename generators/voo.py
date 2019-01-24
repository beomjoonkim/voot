import sys
import numpy as np
sys.path.append('../mover_library/')
from samplers import gaussian_randomly_place_in_region
from generator import Generator
from utils import pick_parameter_distance, place_parameter_distance, se2_distance, visualize_path
from utils import *
from planners.mcts_utils import make_action_executable
import time


class VOOGenerator(Generator):
    def __init__(self, operator_name, problem_env, explr_p, c1):
        Generator.__init__(self, operator_name, problem_env)
        self.explr_p = explr_p
        self.evaled_actions = []
        self.evaled_q_values = []
        self.c1 = c1
        self.feasible_actions = []
        self.feasible_q_values = []
        self.idx_to_update  = None
        self.robot = self.problem_env.robot

    def update_evaled_values(self, node):
        executed_actions_in_node = node.Q.keys()
        executed_action_values_in_node = node.Q.values()
        if len(executed_action_values_in_node) == 0:
            return

        if self.idx_to_update is not None:
            found = False
            for a, q in zip(executed_actions_in_node, executed_action_values_in_node):
                if np.all(np.isclose(self.evaled_actions[self.idx_to_update], make_action_executable(a)['action_parameters'])):
                    found = True
                    break
            try:
                assert found
            except:
                import pdb;pdb.set_trace()
            #assert self.evaled_q_values[self.idx_to_update] == 'update_me'
            self.evaled_q_values[self.idx_to_update] = q

        feasible_idxs = np.where(np.array(executed_action_values_in_node) != self.problem_env.infeasible_reward)[0].tolist()
        assert np.sum(np.array(executed_action_values_in_node) != self.problem_env.infeasible_reward) == len(feasible_idxs)
        for i in feasible_idxs:
            action = executed_actions_in_node[i]
            q_value = executed_action_values_in_node[i]

            executable_action = make_action_executable(action)

            is_in_array = [np.array_equal(executable_action['action_parameters'], a) for a in self.evaled_actions]
            is_action_included = np.any(is_in_array)

            try:
                assert is_action_included
            except:
                import pdb;pdb.set_trace()
            self.evaled_q_values[np.where(is_in_array)[0][0]] = q_value

    def sample_next_point(self, node, n_iter):
        self.update_evaled_values(node)

        rnd = np.random.random() # this should lie outside
        is_sample_from_best_v_region = rnd < 1 - self.explr_p and len(self.evaled_actions) > 1 and \
                                       np.max(node.reward_history.values()) > 0 #self.problem_env.infeasible_reward
        #if node.parent is not None and node.parent.Nvisited > 30:
        #    import pdb;pdb.set_trace()
        if is_sample_from_best_v_region:
            print 'Sample from best region'
        else:
            maxrwd = 0 if len(self.evaled_actions)==0 else np.max(node.reward_history.values())
            print 'Sample from uniform, max rwd: ', maxrwd
        stime=time.time()
        for i in range(n_iter):
            #print i
            if is_sample_from_best_v_region:
                action_parameters = self.sample_from_best_voronoi_region(node)
            else:
                action_parameters = self.sample_from_uniform()
            action, status = self.feasibility_checker.check_feasibility(node,  action_parameters)

            if status == 'HasSolution':
                self.evaled_actions.append(action_parameters)
                self.evaled_q_values.append('update_me')
                self.idx_to_update = len(self.evaled_actions)-1
                #print "Found feasible sample"
                #print "VOO time", time.time()-stime
                break
            else:
                pass
                #self.evaled_q_values.append(self.problem_env.infeasible_reward)
                #self.idx_to_update = None

        return action

    def sample_from_best_voronoi_region(self, node):
        operator = node.operator
        obj = node.obj
        region = node.region
        if operator == 'two_arm_pick':
            params = self.sample_pick_from_best_voroi_region(obj)
        elif operator == 'two_arm_place':
            params = self.sample_place_from_best_voroi_region(node)
        elif operator == 'next_base_pose':
            params = self.sample_place_from_best_voroi_region()
        return params

    def sample_place_from_best_voroi_region(self, node):
        best_dist = np.inf
        other_dists = np.array([-1])
        counter = 0

        best_action_idxs = np.argwhere(self.evaled_q_values == np.amax(self.evaled_q_values))
        best_action_idxs = best_action_idxs.reshape((len(best_action_idxs,)))
        best_action_idx = np.random.choice(best_action_idxs)
        best_evaled_action = self.evaled_actions[best_action_idx]
        other_actions = self.evaled_actions
        # todo closest to any one of the best

        while np.any(best_dist > other_dists) and counter < 1000:
            variance = (self.domain[1] - self.domain[0]) / np.exp(counter)
            new_parameters = np.random.normal(best_evaled_action, variance)
            new_parameters = np.clip(new_parameters, self.domain[0], self.domain[1])
            new_parameters = self.sample_from_uniform()

            best_dist = place_parameter_distance(new_parameters, best_evaled_action, self.c1)
            other_dists = np.array([place_parameter_distance(other, new_parameters, self.c1) for other in other_actions])
            counter += 1
            #print "Gaussian place sampling, variance and counter", variance, counter, len(other_dists)
        #print best_evaled_action
        #best_action, status = self.feasibility_checker.check_feasibility(node, best_evaled_action)
        #action, status = self.feasibility_checker.check_feasibility(node, new_parameters)
        return new_parameters

    def sample_pick_from_best_voroi_region(self, obj):
        best_dist = np.inf
        other_dists = np.array([-1])
        counter = 0

        best_action_idxs = np.argwhere(self.evaled_q_values == np.amax(self.evaled_q_values))
        best_action_idxs = best_action_idxs.reshape((len(best_action_idxs, )))

        best_action_idx = np.random.choice(best_action_idxs)
        best_evaled_action = self.evaled_actions[best_action_idx]
        other_actions = self.evaled_actions
        #if len(best_action_idxs) > 1:
        #    import pdb;pdb.set_trace()
        # todo closest to any one of the best

        while np.any(best_dist > other_dists) and counter < 1000:
            variance = (self.domain[1] - self.domain[0]) / np.exp(counter)
            new_parameters = np.random.normal(best_evaled_action, variance)
            new_parameters = np.clip(new_parameters, self.domain[0], self.domain[1])

            best_dists = [pick_parameter_distance(obj, new_parameters, self.evaled_actions[idx]) for idx in best_action_idxs]
            other_dists = np.array([pick_parameter_distance(obj, other, new_parameters) for other in
                                    other_actions])
            for best_dist in best_dists:
                if np.all(best_dist < other_dists):
                    break
            counter += 1
            #print "Gaussian pick sampling, variance and counter", variance, counter, len(other_dists)

        return new_parameters



