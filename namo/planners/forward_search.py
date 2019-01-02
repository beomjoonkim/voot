import numpy as np
import sys
import time
import copy
from manipulation.constants import PARALLEL_LEFT_ARM, REST_LEFT_ARM, HOLDING_LEFT_ARM, FOLDED_LEFT_ARM, \
    FAR_HOLDING_LEFT_ARM, LOWER_TOP_HOLDING_LEFT_ARM, REGION_Z_OFFSET

sys.path.append('../mover_library/')
from TreeNode import *
from utils import visualize_path, grab_obj, release_obj
from misc.priority_queue import Stack, Queue, PriorityQueue
from manipulation.primitives.savers import DynamicEnvironmentStateSaver
from manipulation.bodies.bodies import set_color
from utils import compute_occ_vec, clean_pose_data, get_body_xytheta, \
    convert_rel_to_abs_base_pose
from data_preprocess.preprocessing_utils import compute_fetch_vec
from data_load_utils import convert_collision_vec_to_one_hot

import numpy as np
import pickle


def create_state(env, current_collisions):
    # state consists of a saver and current collisions
    new_saver = DynamicEnvironmentStateSaver(env)
    new_state = (new_saver, current_collisions)
    new_state_pval = len(current_collisions)
    return new_state, new_state_pval


def check_if_obj_was_maniped(node, curr_obj_name):
    have_placed_this_obj = False
    n = node.parent
    while n is not None:
        touched_obj_name = n.state[1][0]
        if curr_obj_name == touched_obj_name:
            have_placed_this_obj = True
        n = n.parent
    return have_placed_this_obj


def forward_dfs_search(problem, pick_pi, place_pi, max_exp=np.inf, \
                       max_time=np.inf, visualize=False):
    initial_state = problem.problem['initial_state']
    objects = problem.problem['objects']
    all_region = problem.problem['all_region']
    env = problem.problem['env']
    initial_collisions = problem.problem['collided_objs']
    target_obj = problem.problem['target_obj']
    robot_initial_config = problem.problem['robot_initial_config']
    original_path = problem.problem['original_path']  # the original path to the target obj
    obj_shapes = problem.problem['obj_shapes']
    is_unif_policy = pick_pi.__module__.find('Unif') != -1 and place_pi.__module__.find('Unif') != -1

    robot = env.GetRobots()[0]
    leftarm_manip = robot.GetManipulator('leftarm')
    rightarm_manip = robot.GetManipulator('rightarm')
    rightarm_torso_manip = robot.GetManipulator('rightarm_torso')

    queue = PriorityQueue()  # Pop the one with lowest priority value
    init_state_pval = np.inf
    queue.push(init_state_pval, (initial_state, [], None))

    rwd_time_list = []
    nodes = []
    if visualize:
        env.SetViewer('qtcoin')
    # visualize_path(robot,original_path)

    max_n_objs_moved = 0
    n_place_samples = 0
    initial_time = time.time()  # beginning of search
    current_collisions = initial_collisions
    # key_configs = pick_pi.key_configs
    key_configs = pickle.load(open('./key_configs/key_configs.p', 'r'))

    # shouldn't I recompute this everytime? No, I simply want objects to be out of this path
    fvec = compute_fetch_vec(key_configs, original_path, robot, env)
    fvec = convert_collision_vec_to_one_hot(np.array(fvec)[None, :])
    problem.fvec = fvec
    problem.key_configs = key_configs

    while ((len(nodes) < max_exp) and (time.time() - initial_time) < max_time):
        if max_exp != np.inf:
            print len(nodes), max_exp, time.time() - initial_time, ' exped/max_exp'

        state, action, parent = queue.pop()
        node = TreeNode(state,
                        action=action,
                        parent=parent)
        node.goal_node_flag = False
        nodes += [node]  # keep tracks of expanded nodes

        # restore the environment
        saver, current_collisions = state
        saver.Restore()
        problem.curr_obj_name = current_collisions[0]

        fc, misc = problem.get_state_features()

        # get progress on the problem
        n_objs_moved = len(initial_collisions) - len(current_collisions)
        print 'moved ', n_objs_moved
        print len(current_collisions), 'collisions left'

        # compute the objects moved so far
        if n_objs_moved > max_n_objs_moved:
            max_n_objs_moved = n_objs_moved
        time_used = time.time() - initial_time
        rwd_time_list.append([time_used, max_n_objs_moved, len(current_collisions)])

        ### Sample actions
        n_actions_per_state = 3
        place_precond = not np.all(np.isclose(leftarm_manip.GetArmDOFValues(), FOLDED_LEFT_ARM))

        if place_precond is False:
            if visualize:
                problem.visualize_pick_base(pick_pi, fc, misc)
            print "Sampling pick..."
            for _ in range(n_actions_per_state):
                action, is_action_success = problem.sample_feasible_pick(pick_pi, fc, misc)
                if is_action_success:
                    problem.apply_pick_action(action, is_unif_policy)
                    new_state, new_state_pval = create_state(env, current_collisions)
                    queue.push(new_state_pval, (new_state, action, node))  # push subsequent states
                    saver.Restore()
            print "Done sampling pick!"
        else:
            if visualize:
                problem.visualize_placements(place_pi, fc, misc)
            print "Sampling place..."
            grab_obj(robot, env.GetKinBody(problem.curr_obj_name))  # restoring breaks contacts
            for _ in range(n_actions_per_state):
                action, is_action_success = problem.sample_feasible_place(place_pi, fc, misc)
                if is_action_success:
                    problem.apply_place_action(action)
                    current_collisions = problem.compute_obj_collisions()
                    is_solution = len(current_collisions) == 0
                    if is_solution:
                        print "Success!"
                        time_used = time.time() - initial_time
                        rwd_time_list.append([time_used, max_n_objs_moved, len(current_collisions)])
                        return nodes, rwd_time_list
                    new_state, new_state_pval = create_state(env, current_collisions)
                    queue.push(new_state_pval, (new_state, action, node))  # push subsequent states
                    saver.Restore()
                    grab_obj(robot, env.GetKinBody(problem.curr_obj_name))
            print "Done sampling place!"
        if queue.empty():
            print "Going back to initial state"
            queue.push(init_state_pval, (initial_state, [], None))

            # Save traj list
    return nodes, rwd_time_list

# Restore:
#   - poses of objects  
#   - grasped object
#   - collision and current object
