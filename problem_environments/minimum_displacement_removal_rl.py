from minimum_displacement_removal import MinimumDisplacementRemoval
from mover_library.utils import get_body_xytheta
from generators.presampled_pick_generator import PreSampledPickGenerator
from mover_library.utils import CustomStateSaver
from planners.mcts_tree_node import TreeNode
from generators.learned_generators.learned_policy_based_generator import LearnedPolicyBasedGenerator
from generators.uniform import UniformGenerator
import numpy as np
import pickle


class RLMinimumDisplacementRemoval(MinimumDisplacementRemoval):
    def __init__(self, problem_idx):
        MinimumDisplacementRemoval.__init__(self, problem_idx)
        swept_volume_to_clear_obstacles_from = self.load_swept_volume()
        initial_collisions = self.get_objs_in_collision(swept_volume_to_clear_obstacles_from, 'entire_region')
        self.initial_collisions = initial_collisions
        self.set_objects_not_in_goal(initial_collisions)
        self.set_swept_volume(swept_volume_to_clear_obstacles_from)

    def load_swept_volume(self):
        swept_volume_file_name = './problem_environments/mover_domain_problems/fetching_path_1.pkl'
        return pickle.load(open(swept_volume_file_name, 'r'))

    def get_state(self):
        obj_poses = np.array([get_body_xytheta(o) for o in self.initial_collisions])
        obj_poses = obj_poses.reshape((1, 7 * 3))
        return obj_poses

    def rollout_the_policy(self, policy, time_step_limit, visualize=False):
        self.placements = []
        states = []
        actions = []
        rewards = []

        pick_generator = UniformGenerator('two_arm_pick', self)
        place_generator = LearnedPolicyBasedGenerator('two_arm_place', self, policy)
        # Which generator to use?
        parent_action = None
        parent_reward = 0

        while len(actions) < time_step_limit:
            curr_state = self.get_state()

            operator_skeleton = self.get_applicable_op_skeleton(parent_action)
            node = TreeNode(operator_skeleton, None, None, None, None, None, None)
            node.parent_action_reward = parent_reward

            is_time_to_pick = 'pick' in operator_skeleton.type
            if is_time_to_pick:
                op_cont_params = pick_generator.sample_next_point(node, n_iter=50)
            else:
                op_cont_params = place_generator.sample_next_point(node, curr_state, n_iter=50)
            operator_skeleton.continuous_parameters = op_cont_params
            action = operator_skeleton
            reward = self.apply_operator_instance(action, node)

            parent_action = action
            parent_reward = reward

            if not is_time_to_pick:
                actions.append(action)  # action performed in current state
                states.append(curr_state)  # konf while holding the object
                rewards.append(reward)

            if reward == -2:
                break  # end of episode

            if len(self.objects_currently_not_in_goal) == 0:
                # reset the environment and the
                self.init_saver.Restore()
                self.objects_currently_not_in_goal = self.objects

        traj = {'s': states, 'a': actions, 'r': rewards}
        return traj
