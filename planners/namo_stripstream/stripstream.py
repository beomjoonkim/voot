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
import pickle


def pklsave(obj, name=''):
    pickle.dump(obj, open('ss_tmp'+str(name)+'.pkl', 'wb'))


def pklload(name=''):
    return pickle.load(open('ss_tmp'+str(name)+'.pkl', 'r'))

def gen_grasp(pick_unif):
    # note generate grasp, ik solution gc, relative base conf, and absolute base transform for grasping
    def fcn(obj_name):
        pick_unif.problem_env.reset_to_init_state_stripstream()
        obj = pick_unif.problem_env.env.GetKinBody(obj_name)

        while True:
            # todo: disable all of objects
            print "Calling gengrasp"
            pick_unif.problem_env.disable_objects_in_region('entire_region')
            obj.Enable(True)
            action = pick_unif.predict(obj, pick_unif.problem_env.regions['entire_region'])
            pick_unif.problem_env.enable_objects_in_region('entire_region')
            pick_base_pose = action['base_pose']
            grasp = action['grasp_params']
            g_config = action['g_config']
            pick_unif.problem_env.reset_to_init_state_stripstream()
            if g_config is None:
                yield None
            else:
                print grasp, pick_base_pose, g_config
                yield [grasp, pick_base_pose, g_config]
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
    def fcn(obj_name, grasp, pick_base_pose, g_config, region_name):
        # simulate pick
        while True:
            obj = problem.env.GetKinBody(obj_name)
            problem.apply_two_arm_pick_action_stripstream((pick_base_pose, grasp, g_config), obj) # how do I ensure that we are in the same state in both openrave and stripstream?
            print region_name

            problem.disable_objects_in_region('entire_region')
            obj.Enable(True)
            place_action = place_unif.predict(obj, problem.regions[region_name])
            place_base_pose = place_action['base_pose']
            object_pose = place_action['object_pose'].squeeze()
            path, status = problem.get_base_motion_plan(place_base_pose.squeeze())
            problem.enable_objects_in_region('entire_region')
            print "Input", obj_name, grasp, pick_base_pose

            problem.reset_to_init_state_stripstream()
            if status == 'HasSolution':
                yield (place_base_pose, object_pose, path)
            else:
                yield None
    return fcn


def check_traj_collision_with_object(problem):
    def fcn(holding_obj_name, grasp, pick_base_conf, g_config, placed_obj_name, placed_obj_pose, q_goal, hodlinig_obj_path):
        holding_obj = problem.env.GetKinBody(holding_obj_name)
        placed_obj = problem.env.GetKinBody(placed_obj_name)
        problem.apply_two_arm_pick_action_stripstream((pick_base_conf, grasp, g_config), holding_obj)  # how do I ensure that we are in the same state in both openrave and stripstream?

        import pdb;pdb.set_trace()
        if np.all(pick_base_conf == q_goal):
            return False

        if holding_obj_name != placed_obj_name:
            # set the obstacle in place
            set_obj_xytheta(placed_obj_pose, placed_obj)
        else:
            return False  # this is already checked

        # check collision
        for p in hodlinig_obj_path:
            set_robot_config(p, problem.robot)
            if problem.env.CheckCollision(problem.robot):
                problem.reset_to_init_state_stripstream()
                return True

        problem.reset_to_init_state_stripstream()
        return False
    return fcn


def gen_base_traj_with_object(problem):
    # note generate object placement, relative base conf, absolute base conf, and the path from q1 to abs base conf
    def fcn(obj_name, grasp, pick_base_pose, g_config, q_init, q_goal):
        # simulate pick
        import pdb;pdb.set_trace()
        while True:
            obj = problem.env.GetKinBody(obj_name)
            problem.disable_objects_in_region('entire_region')
            obj.Enable(True)
            problem.apply_two_arm_pick_action_stripstream((pick_base_pose, grasp, g_config), obj) # how do I ensure that we are in the same state in both openrave and stripstream?
            set_robot_config(q_init, problem.robot)
            path, status = problem.get_base_motion_plan(q_goal.squeeze())
            problem.enable_objects_in_region('entire_region')

            if np.all(q_init == q_goal):
                return ([q_init],)

            if status == "HasSolution":
                problem.reset_to_init_state_stripstream()
                print "Input", obj_name, grasp, pick_base_pose
                # ('obj0', array([2.6868955, 0.64292839, 0.22429731]), array([4.41584923, 1.52569695, -2.24756557]))
                return (path,)
            else:
                problem.reset_to_init_state_stripstream()
                return None
    return fcn


def gen_base_traj(namo):
    def fcn(q_init, q_goal):
        while True:
            if np.all(q_init == q_goal):
                yield ([q_init],)
            curr_robot_config = get_body_xytheta(namo.robot)
            set_robot_config(q_init, namo.robot)
            namo.disable_objects_in_region('entire_region')
            plan, status = namo.get_base_motion_plan(q_goal)
            namo.enable_objects_in_region('entire_region')
            set_robot_config(curr_robot_config, namo.robot)
            if status == 'HasSolution':
                yield (plan,)
            else:
                yield None
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
                  'TrajPoseCollisionWithObject': check_traj_collision_with_object(namo),
                  'gen-base-traj': from_gen_fn(gen_base_traj(namo)),
                  'gen-placement': from_gen_fn(gen_placement(namo, place_sampler)),
                  #'gen-base-traj-with-obj': from_fn(gen_base_traj_with_object(namo)),
                  }
    obj_names = problem_config['obj_poses'].keys()
    obj_poses = problem_config['obj_poses'].values()
    init = [('Pickable', obj_name) for obj_name in obj_names]
    init += [('Robot', 'pr2')]
    init += [('EmptyArm',)]
    init += [('Region', 'entire_region')]
    init += [('Region', 'loading_region')]
    #init += [('InRegion', 'obj0', 'entire_region')]
    #init += [('InRegion', 'obj1', 'entire_region')]
    init += [('AtPose', obj_name, obj_pose) for obj_name, obj_pose in zip(obj_names, obj_poses)]
    init += [('Pose', obj_name, obj_pose) for obj_name, obj_pose in zip(obj_names, obj_poses)]

    init_config = np.array([-1, 1, 0])
    init += [('BaseConf', init_config)]
    init += [('AtConf', init_config)]

    goal = ['and', ('InRegion', 'obj0', 'loading_region')]
    #goal = ['and', ('AtConf', goal_config), ('not', ('EmptyArm', ))]
    return (domain_pddl, constant_map, stream_pddl, stream_map, init, goal), namo

##################################################

def process_plan(plan, namo):
    namo.env.SetViewer('qtcoin')
    namo.reset_to_init_state_stripstream()
    for step_idx, step in enumerate(plan):
        # todo finish this visualization script
        import pdb;pdb.set_trace()
        if step[0] == 'pickup':
            obj_name = step[1][0]
            grasp = step[1][1]
            pick_base_pose = step[1][2]
            g_config = step[1][3]
            namo.apply_two_arm_pick_action_stripstream((pick_base_pose, grasp, g_config), namo.env.GetKinBody(obj_name))
        elif step[0] == 'movebase':
            q_init = step[1][0]
            q_goal = step[1][1]
            path = step[1][2]
            visualize_path(namo.robot, path)
            set_robot_config(q_goal, namo.robot)
        elif step[0] == 'movebase_with_object':
            q_init = step[1][4]
            q_goal = step[1][5]
            path = step[1][6]
            set_robot_config(q_init, namo.robot)
            visualize_path(namo.robot, path)
            set_robot_config(q_goal, namo.robot)
        else:
            place_obj_name = step[1][0]
            place_base_pose = step[1][4]
            path = step[1][-1]
            visualize_path(namo.robot, path)
            action = {'base_pose': place_base_pose}
            obj = namo.env.GetKinBody(place_obj_name)
            two_arm_place_object(obj, namo.robot, action)


def solve_pddlstream():
    pddlstream_problem, namo = get_problem()
    namo.env.SetViewer('qtcoin')
    stime = time.time()
    #solution = solve_incremental(pddlstream_problem, unit_costs=True, max_time=500)
    solution = solve_focused(pddlstream_problem, unit_costs=True, max_time=500)
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


