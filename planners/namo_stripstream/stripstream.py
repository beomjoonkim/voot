import os
import time
import sys

from pddlstream.algorithms.focused import solve_focused
from pddlstream.algorithms.incremental import solve_incremental
from pddlstream.utils import print_solution, read
from pddlstream.language.generator import from_gen_fn, from_test, fn_from_constant, from_fn

from problem_environments.stripstream_namo_env import StripStreamNAMO
from generators.PickUniform import PickWithBaseUnif
from generators.PlaceUniform import PlaceUnif

sys.path.append('../mover_library/')
from utils import compute_occ_vec, set_robot_config, remove_drawn_configs, \
    draw_configs, clean_pose_data, draw_robot_at_conf, set_active_dof_conf, get_body_xytheta, \
    check_collision_except, release_obj, draw_robot_base_configs, two_arm_place_object, visualize_path, set_obj_xytheta

import numpy as np
import openravepy


def gen_grasp(pick_unif):
    # note generate grasp, ik solution gc, relative base conf, and absolute base transform for grasping
    def fcn(obj_name):
        pick_unif.problem_env.reset_to_init_state_stripstream()
        obj = pick_unif.problem_env.env.GetKinBody(obj_name)

        while True:
            # todo: disable all of objects
            print "Calling gengrasp"
            action = pick_unif.predict(obj, pick_unif.problem_env.regions['entire_region'])
            pick_base_pose = action['base_pose']
            grasp = action['grasp_params']
            g_config = action['g_config']
            pick_unif.problem_env.reset_to_init_state_stripstream()
            if g_config is None:
                yield None
            print grasp, pick_base_pose
            yield [grasp, pick_base_pose]
    return fcn


def check_traj_collision(problem):
    def fcn(obstacle_name, obstacle_pose, q_init, q_goal, traj):
        obstacle = problem.env.GetKinBody(obstacle_name)
        set_obj_xytheta(obstacle_pose, obstacle)

        # check collision
        for p in traj:
            set_robot_config(p, problem.robot)
            if problem.env.CheckCollision(problem.robot):
                problem.reset_to_init_state_stripstream()
                return True

        problem.reset_to_init_state_stripstream()
        return False

    return fcn


def gen_placement(problem, place_unif):
    # note generate object placement, relative base conf, absolute base conf, and the path from q1 to abs base conf
    def fcn(obj_name, grasp, pick_base_pose, region_name):
        # simulate pick
        import pdb;pdb.set_trace()
        while True:
            obj = problem.env.GetKinBody(obj_name)
            problem.apply_two_arm_pick_action_stripstream((pick_base_pose, grasp), obj) # how do I ensure that we are in the same state in both openrave and stripstream?
            place_action = place_unif.predict(obj, problem.regions['object_region'])
            place_base_pose = place_action['base_pose']
            object_pose = place_action['object_pose'].squeeze()

            assert obj.IsEnabled()
            assert problem.env.GetKinBody('floorwalls').IsEnabled()
            assert len(problem.robot.GetGrabbed()) != 0
            assert problem.robot.IsEnabled()

            problem.reset_to_init_state_stripstream()
            print "Input", obj_name, grasp, pick_base_pose
            return (place_base_pose, object_pose)

    return fcn


def gen_base_traj(namo):
    def fcn(q_init, q_goal):
        while True:
            if np.all(q_init == q_goal):
                return ([q_init],)
            curr_robot_config = get_body_xytheta(namo.robot)
            set_robot_config(q_init, namo.robot)
            plan, status = namo.get_base_motion_plan(q_goal)
            set_robot_config(curr_robot_config, namo.robot)
            if status == 'HasSolution':
                return (plan,)
            else:
                return None
    return fcn

def read_pddl(filename):
    directory = os.path.dirname(os.path.abspath(__file__))
    return read(os.path.join(directory, filename))


def get_problem():
    namo = StripStreamNAMO()
    problem_config = namo.problem_config
    directory = os.path.dirname(os.path.abspath(__file__))
    domain_pddl = read(os.path.join(directory, 'domain.pddl'))
    stream_pddl = read(os.path.join(directory, 'stream.pddl'))

    pick_sampler = PickWithBaseUnif(namo)
    place_sampler = PlaceUnif(namo)
    constant_map = {}
    stream_map = {'gen-grasp': from_gen_fn(gen_grasp(pick_sampler)),
                  'TrajPoseCollision': check_traj_collision(namo),
                  'gen-base-traj': from_fn(gen_base_traj(namo)),
                  'gen-placement': from_fn(gen_placement(namo, place_sampler))
                  }
    obj_names = problem_config['obj_poses'].keys()
    obj_poses = problem_config['obj_poses'].values()
    init = [('Pickable', obj_name) for obj_name in obj_names]
    init += [('Robot', 'pr2')]
    init += [('EmptyArm',)]
    init += [('Region', 'entire_region')]
    init += [('Region', 'loading_region')]
    init += [('InRegion', 'obj0', 'entire_region')]
    init += [('InRegion', 'obj1', 'entire_region')]
    init += [('AtPose', obj_name, obj_pose) for obj_name, obj_pose in zip(obj_names, obj_poses)]
    init += [('Pose', obj_name, obj_pose) for obj_name, obj_pose in zip(obj_names, obj_poses)]

    init_config = np.array([-1, 1, 0])
    init += [('BaseConf', init_config)]
    init += [('AtConf', init_config)]

    goal = ['and', ('InRegion', 'obj0', 'loading_region')]
    return (domain_pddl, constant_map, stream_pddl, stream_map, init, goal), namo

##################################################

def process_plan(plan, namo):
    namo.env.SetViewer('qtcoin')
    namo.reset_to_init_state_stripstream()
    for step_idx, step in enumerate(plan):
        # todo finish this visualization script
        if step[0] == 'pickup':
            obj_name = step[1][0]
            grasp = step[1][1]
            pick_base_pose = step[1][2]
            namo.apply_two_arm_pick_action((pick_base_pose, grasp), namo.env.GetKinBody(obj_name))
        elif step[0] == 'movebase':
            q_init = step[1][0]
            q_goal = step[1][1]
            path = step[1][2]
            visualize_path(namo.robot, path)
            set_robot_config(q_goal, namo.robot)
        else:
            place_obj_name = step[1][0]
            place_base_pose = step[1][4]
            path = step[1][5]
            visualize_path(namo.robot, path)
            action = {'base_pose': place_base_pose}
            obj = namo.env.GetKinBody(place_obj_name)
            two_arm_place_object(obj, namo.robot, action)


def solve_pddlstream():
    pddlstream_problem, namo = get_problem()
    stime = time.time()
    solution = solve_incremental(pddlstream_problem, unit_costs=True, max_time=500)
    #solution = solve_focused(pddlstream_problem, unit_costs=True, max_time=500)
    search_time = time.time()-stime
    plan, cost, evaluations = solution
    print "Search time", search_time
    import pdb;pdb.set_trace()
    if solution[0] is None:
        print "No Solution"
        sys.exit(-1)
    print_solution(solution)
    process_plan(plan, namo)
    namo.problem_config['env'].Destroy()
    openravepy.RaveDestroy()

    return plan, search_time

##################################################


