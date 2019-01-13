import sys
import numpy as np
sys.path.append('../mover_library/')
from generator import Generator
from utils import pick_parameter_distance, place_parameter_distance, convert_base_pose_to_se2, convert_se2_to_base_pose
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

        if operator_name == 'two_arm_pick':
            dim_x = 6
        else:
            dim_x = 3
        self.gp = StandardContinuousGP(dim_x)
        self.acq_fcn = UCB(zeta=explr_p, gp=self.gp)
        self.gp_format_domain = Domain(0, self.domain)
        self.gp_optimizer = BO(self.gp, self.acq_fcn, self.gp_format_domain)  # this depends on the problem

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
            if i > n_iter*0.5 or len(self.evaled_actions) == 0:
                #print "GP sampling from uniform", n_iter*0.5, n_iter, len(self.evaled_actions)
                action_parameters = self.sample_from_uniform()
            else:
                #print "GP sampling from GP"
                action_parameters = self.choose_next_point(node)

            action, status = self.feasibility_checker.check_feasibility(node,  action_parameters)

            if status == 'HasSolution':
                print "Found feasible sample"
                break
            else:
                self.evaled_actions.append(action_parameters)
                self.evaled_q_values.append(self.problem_env.infeasible_reward)
        return action

    def choose_next_point(self, node):
        if node.operator == 'two_arm_place':
            evaled_actions_in_se2 = self.evaled_actions
            next_point = self.gp_optimizer.choose_next_point(evaled_actions_in_se2, self.evaled_q_values)
        else:
            next_point = self.gp_optimizer.choose_next_point(self.evaled_actions, self.evaled_q_values)
        return next_point




