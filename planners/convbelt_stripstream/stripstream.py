import os
import time
import sys

from pddlstream.algorithms.focused import solve_focused
from pddlstream.algorithms.incremental import solve_incremental
from pddlstream.utils import print_solution, read
from pddlstream.language.generator import from_gen_fn, from_test, fn_from_constant, from_fn

from problem_environments.conveyor_belt_env import ConveyorBelt
from generators.PickUniform import PickWithBaseUnif
from generators.PlaceUniform import PlaceUnif

sys.path.append('../mover_library/')
from utils import compute_occ_vec, set_robot_config, remove_drawn_configs, \
    draw_configs, clean_pose_data, draw_robot_at_conf, set_active_dof_conf, get_body_xytheta, \
    check_collision_except, release_obj, draw_robot_base_configs, two_arm_place_object, visualize_path, set_obj_xytheta

from manipulation.primitives.transforms import quat_from_z_rot, set_point, set_quat

import numpy as np
import openravepy


def gen_grasp(pick_unif):
    # note generate grasp, ik solution gc, relative base conf, and absolute base transform for grasping
    def fcn(obj_name):
        pick_unif.problem_env.reset_to_init_state_stripstream()
        obj = pick_unif.problem_env.env.GetKinBody(obj_name)

        while True:
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
    def fcn(holding_obj_name, grasp, pick_base_conf, placed_obj_name, placed_obj_pose, holding_obj_place_base_pose, holding_obj_place_traj):
        holding_obj = problem.env.GetKinBody(holding_obj_name)
        placed_obj = problem.env.GetKinBody(placed_obj_name)
        problem.apply_two_arm_pick_action_stripstream((pick_base_conf, grasp), holding_obj)

        if holding_obj_name != placed_obj_name:
            # set the obstacle in place
            set_obj_xytheta(placed_obj_pose, placed_obj)
        else:
            return False # this is already checked

        if len(problem.robot.GetGrabbed()) == 0:
            import pdb;pdb.set_trace()

        # check collision
        for p in holding_obj_place_traj:
            set_robot_config(p, problem.robot)
            if problem.env.CheckCollision(problem.robot):
                problem.reset_to_init_state_stripstream()
                return True

        problem.reset_to_init_state_stripstream()
        return False

    return fcn


def gen_placement(problem, place_unif):
    # note generate object placement, relative base conf, absolute base conf, and the path from q1 to abs base conf
    def fcn(obj_name, grasp, pick_base_pose):
        # simulate pick
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

            path, status = problem.get_base_motion_plan(place_base_pose.squeeze())

            if status == "HasSolution":
                problem.reset_to_init_state_stripstream()
                print "Input", obj_name, grasp, pick_base_pose
                # ('obj0', array([2.6868955, 0.64292839, 0.22429731]), array([4.41584923, 1.52569695, -2.24756557]))
                return (place_base_pose, object_pose, path)
            else:
                problem.reset_to_init_state_stripstream()
                return None

    return fcn


def read_pddl(filename):
    directory = os.path.dirname(os.path.abspath(__file__))
    return read(os.path.join(directory, filename))


def get_problem():
    convbelt = ConveyorBelt()
    problem_config = convbelt.problem_config
    directory = os.path.dirname(os.path.abspath(__file__))
    domain_pddl = read(os.path.join(directory, 'domain.pddl'))
    stream_pddl = read(os.path.join(directory, 'stream.pddl'))

    pick_sampler = PickWithBaseUnif(convbelt)
    place_sampler = PlaceUnif(convbelt)
    constant_map = {}
    stream_map = {'gen-grasp': from_gen_fn(gen_grasp(pick_sampler)),
                  'gen-placement': from_fn(gen_placement(convbelt, place_sampler)),
                  #'TrajPoseCollision': fn_from_constant(False)
                  'TrajPoseCollision': check_traj_collision(convbelt),
                  }
    obj_names = problem_config['obj_poses'].keys()
    obj_poses = problem_config['obj_poses'].values()
    obj_names = ['obj0', 'obj1', 'obj2', 'obj3', 'obj4']
    init = [('Pickable', obj_name) for obj_name in obj_names]
    init += [('Robot', 'pr2')]
    init += [('BaseConf', [0, 1.05, 0])]
    init += [('EmptyArm',)]
    init += [('ObjectZero', 'obj0')]
    init += [('ObjectOne', 'obj1')]
    init += [('ObjectTwo', 'obj2')]
    init += [('ObjectThree', 'obj3')]
    init += [('ObjectFour', 'obj4')]

    #goal = ('Picked', 'obj0')
    #goal = ('AtPose', 'obj0', obj_pose)
    goal = ['and', ('InLoadingRegion', 'obj0'),
                   ('InLoadingRegion', 'obj1'),
                   ('InLoadingRegion', 'obj2'),
                   ('InLoadingRegion', 'obj3'),
                   ('InLoadingRegion', 'obj4')]
    #goal = ['and', ('InLoadingRegion', 'obj0'),  ('InLoadingRegion', 'obj1')]

    #goal = ('InLoadingRegion', 'obj0')
    convbelt.env.SetViewer('qtcoin')
    return (domain_pddl, constant_map, stream_pddl, stream_map, init, goal), convbelt

##################################################
def process_plan(plan, convbelt):
    convbelt.env.SetViewer('qtcoin')
    convbelt.reset_to_init_state_stripstream()
    for step_idx, step in enumerate(plan):
        # todo finish this visualization script
        if step[0].find('pickup')!=-1:
            obj_name = step[1][0]
            grasp = step[1][1]
            pick_base_pose = step[1][2]
            convbelt.apply_two_arm_pick_action_stripstream((pick_base_pose, grasp), convbelt.env.GetKinBody(obj_name))
        else:
            place_obj_name = step[1][0]
            place_base_pose = step[1][4]
            path = step[1][5]
            visualize_path(convbelt.robot, path)
            action = {'base_pose': place_base_pose}
            obj = convbelt.env.GetKinBody(place_obj_name)
            two_arm_place_object(obj, convbelt.robot, action)


def solve_pddlstream():
    pddlstream_problem, convbelt = get_problem()
    stime = time.time()
    #solution = solve_incremental(pddlstream_problem, unit_costs=True, max_time=np.inf)
    solution = solve_focused(pddlstream_problem, unit_costs=True, max_time=300)
    search_time = time.time()-stime
    plan, cost, evaluations = solution
    print "Search time", search_time
    """
    if solution[0] is None:
        sys.exit(-1)
    print_solution(solution)
    process_plan(plan, convbelt)
    import pdb;pdb.set_trace()
    """
    convbelt.problem_config['env'].Destroy()
    openravepy.RaveDestroy()

    return plan, search_time

##################################################


