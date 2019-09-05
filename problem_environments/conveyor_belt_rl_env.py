from conveyor_belt_env import ConveyorBelt
from mover_library.utils import get_body_xytheta
from generators.presampled_pick_generator import PreSampledPickGenerator
from mover_library.utils import CustomStateSaver
from planners.mcts_tree_node import TreeNode
from generators.learned_generators.learned_policy_based_generator import LearnedPolicyBasedGenerator
import numpy as np
from mover_library import utils


class RLConveyorBelt(ConveyorBelt):
    def __init__(self, problem_idx, n_actions_per_node):
        ConveyorBelt.__init__(self, problem_idx, n_actions_per_node)
        self.set_objects_not_in_goal(self.objects)

    def get_state(self):
        # return the poses of objects
        obj_poses = np.array([get_body_xytheta(o) for o in self.objects])
        obj_poses = obj_poses.reshape((1, 20 * 3))
        return obj_poses

    def rollout_the_policy(self, policy, time_step_limit, visualize=False):
        self.placements = []
        states = []
        actions = []
        rewards = []

        pick_generator = PreSampledPickGenerator()
        place_generator = LearnedPolicyBasedGenerator('3_paps', self, policy)
        parent_action = None

        while len(actions) < time_step_limit:
            curr_state = self.get_state()
            operator_skeleton = self.get_applicable_op_skeleton(parent_action)

            node = TreeNode(operator_skeleton, None, None, None, None, None, None)

            is_time_to_pick = 'pick' in operator_skeleton.type
            if is_time_to_pick:
                op_cont_params = pick_generator.sample_next_point(node, n_iter=50)
            else:
                op_cont_params = place_generator.sample_next_point(node, curr_state, n_iter=50)
            operator_skeleton.continuous_parameters = op_cont_params
            action = operator_skeleton
            reward = self.apply_operator_instance(action, node=None)
            parent_action = action

            if not is_time_to_pick:
                actions.append(action)  # action performed in current state
                states.append(curr_state)  # konf while holding the object
                rewards.append(reward)

            if reward == -2:
                break

            if len(self.objects_currently_not_in_goal) == 0:
                # reset the environment and the
                self.init_saver.Restore()
                self.objects_currently_not_in_goal = self.objects

        traj = {'s': states, 'a': actions, 'r': rewards}
        return traj
