# search related libs
from TreeNode import *
from generators.PlaceUniform import PlaceUnif
from misc.priority_queue import Stack, Queue, FILOPriorityQueue, PriorityQueue

import sys

sys.path.append('../mover_library/')
from samplers import *
from utils import *

import time


def sample_action(conv_belt, policy):
    env = conv_belt.problem['env']
    robot = env.GetRobots()[0]
    loading_region = conv_belt.problem['loading_region']
    all_region = conv_belt.problem['all_region']

    is_unif_policy = policy.__module__.find("Unif") != -1
    if is_unif_policy:
        for _ in range(100):
            place_robot_pose = policy.predict(conv_belt.curr_obj)[0]
            place_robot_pose = place_robot_pose.reshape((1, 3))
            is_feasible = conv_belt.check_feasible_base_pose(place_robot_pose)
            if is_feasible:
                break
    else:
        # create features
        # get actions
        state = conv_belt.get_state(policy.key_configs)
        for _ in range(100):
            place_robot_pose = policy.predict(state)
            is_feasible = conv_belt.check_feasible_base_pose(place_robot_pose)
            if is_feasible:
                break
    if is_feasible:
        return place_robot_pose
    else:
        return None


def compute_V(placements, conv_belt, policy):
    OBJECTS = conv_belt.problem['objects']
    is_unif_policy = policy.__module__.find("Unif") != -1

    if is_unif_policy:
        v = -(len(OBJECTS) - len(placements))  # remaning objs to be packed
    else:
        state = conv_belt.get_state(policy.key_configs)
        v = policy.predict_V(state)

    return v


def add_node(nodes, state, sample, parent, rwd):
    node = TreeNode(state, \
                    sample=sample, \
                    parent=parent, \
                    rwd=rwd)
    node.goal_node_flag = False
    node.pred_time = 0
    nodes += [node]
    return node


def create_new_state(env, placements):
    new_saver = DynamicEnvironmentStateSaver(env)
    new_state = (new_saver, placements)  # collisions are preserved
    return new_state


def forward_search(conv_belt, \
                   max_exp, \
                   policy):
    problem = conv_belt.problem
    initial_state = problem['initial_state']
    OBSTACLES = problem['obstacles']
    OBJECTS = problem['objects']
    loading_region = problem['loading_region']
    all_region = problem['all_region']
    env = problem['env']

    robot = env.GetRobots()[0]
    leftarm_manip = robot.GetManipulator('leftarm')
    rightarm_manip = robot.GetManipulator('rightarm')
    rightarm_torso_manip = robot.GetManipulator('rightarm_torso')

    initial_time = time.time()
    max_placements = 0
    init_base_conf = np.array([0, 1.05, 0])
    robot = env.GetRobots()[0]

    queue = PriorityQueue()
    init_state_pval = len(OBJECTS) + 1  # does it prefer the lower values?
    queue.push(init_state_pval, (initial_state, None, None))  # TODO - put the nodes back in

    # number of objects placed after using x amount of time
    rwd_n_expanded_list = []
    nodes = []
    goal_state, last_node = None, None
    max_placements = 0

    while (goal_state is None and not queue.empty()) \
            and (len(nodes) < max_exp):
        # print times
        if max_exp != np.inf:
            print len(nodes), max_exp, time.time() - initial_time, ' exped/max_exp,time_used'

        state, sample, parent = queue.pop()
        saver, placements = state

        curr_node = add_node(nodes, state, sample, parent, len(placements))

        # restore the environment
        saver.Restore()

        if max_placements < len(placements):
            max_placements = len(placements)
        print max_placements, 'rwd'
        rwd_n_expanded_list.append([time.time() - initial_time, max_placements])

        # sample K actions
        n_actions_per_state = 3
        n_actions = 0
        conv_belt.curr_obj = OBJECTS[len(placements)]  # fixed object order

        # time to place if my arms are not folded
        place_precond = not np.all(np.isclose(leftarm_manip.GetArmDOFValues(), FOLDED_LEFT_ARM))
        if place_precond:
            if conv_belt.v:
                saver.Restore()
                grab_obj(robot, conv_belt.curr_obj)
                conv_belt.visualize_placements(policy)

            for ntry in range(n_actions_per_state):
                saver.Restore()
                grab_obj(robot, conv_belt.curr_obj)

                place_robot_pose = sample_action(conv_belt, policy)
                if place_robot_pose is None:
                    continue
                has_path = conv_belt.check_action_feasible(place_robot_pose)

                place = {}
                if has_path:
                    place['place_base_pose'] = place_robot_pose
                    place['obj'] = conv_belt.curr_obj.GetName()

                    set_robot_config(place_robot_pose, robot)
                    place_obj(conv_belt.curr_obj, robot, FOLDED_LEFT_ARM, leftarm_manip, rightarm_manip)
                    set_robot_config(init_base_conf, robot)

                    new_placements = placements + [place_robot_pose]
                    print 'New placements', new_placements
                    new_state = create_new_state(env, new_placements)
                    is_goal = len(new_placements) == len(OBJECTS)
                    if is_goal:
                        import pdb;
                        pdb.set_trace()
                        print "Success"
                        add_node(nodes, new_state, place, curr_node, len(OBJECTS))
                        rwd_n_expanded_list.append([len(nodes), len(OBJECTS)])
                        return nodes, rwd_n_expanded_list

                    # new_state_val = -compute_V( new_placements,conv_belt,policy ) # smaller the better
                    new_state_val = -(len(OBJECTS) - len(new_placements))  # remaning objs to be packed
                    new_state_val = -(len(new_placements))  # remaning objs to be packed
                    print 'New state value is ', new_state_val
                    queue.push(new_state_val, (new_state, place, curr_node))  # push subsequent states
        else:
            for ntry in range(n_actions_per_state):
                saver.Restore()
                pick = conv_belt.apply_pick_action()

                new_state = create_new_state(env, placements)
                # new_state_val = -compute_V( placements,conv_belt,policy ) # smaller the better
                new_state_val = -(len(OBJECTS) - len(placements))  # remaning objs to be packed
                new_state_val = -(len(placements))  # remaning objs to be packed
                print "Pick new state val", new_state_val
                queue.push(new_state_val, (new_state, pick, curr_node))  # push subsequent states

        if queue.empty():
            print "Persistency..."
            # persistency
            queue.push(init_state_pval, (initial_state, None, None))

            # What's the intuition in backpropagatig the values in AlphaZero?
    #  - At the end of the day, you want to return the action with maximum value
    #  - This value gets accurate as you do more roll-outs, and as you get near the end
    #  - of the game, because in the game of Go, your reward is 0 or 1 at the end
    # I probably don't need such update. All I am trying to do is to
    # find a path to the goal. For the game of Go, we cannot know if we are at the
    # winning state - or a terminal state even, I think.
    # But can this help in getting a better estimate of the heuristic?
    # They do not have any heuristic at all, but we do.

    # The question is if the value function that we learn is better than
    # the heuristic function.

    # heuristic function = T - n_objs_packed
    # Q fcn approximates sumR from the current state

    # which one is more accurate in terms of
    # getting the state that is closer to the goal?

    # The heuristic function can be wrong if we are at the deadend
    # On the other hand, if we are at the deadend, Q function can detect that

    # Heuristic function is local; Q fcn is global

    # What if I do 1 and 0 for scoring trajectories?
    # Then it will be sample inefficient because
    # it will ignore the ones with that packed 4 objects
    # But this depends on how I defined "success"
    # If my goal is to maximize the number of objects packed, the current
    # reward function is correct. But if my goal is to pack more than 4 objects,
    # then I can do 0/1 reward on trajectories
    return nodes, rwd_n_expanded_list
