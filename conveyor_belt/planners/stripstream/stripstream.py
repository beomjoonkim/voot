import os
import time

from pddlstream.algorithms.focused import solve_focused
from pddlstream.algorithms.incremental import solve_incremental
from pddlstream.utils import print_solution, read
from pddlstream.language.generator import from_gen_fn, from_test, fn_from_constant, from_fn
from conveyor_belt_env import ConveyorBelt
from generators.PickUniform import PickUnif
from generators.PlaceUniform import PlaceUnif


import sys
import numpy as np
sys.path.append('../mover_library/')
from utils import compute_occ_vec, set_robot_config, remove_drawn_configs, \
    draw_configs, clean_pose_data, draw_robot_at_conf, set_active_dof_conf, get_body_xytheta, \
    pick_obj, place_obj, check_collision_except, release_obj, draw_robot_base_configs

from manipulation.primitives.transforms import quat_from_z_rot, set_point, set_quat


def gen_grasp(pick_unif):
    # note generate grasp, ik solution gc, relative base conf, and absolute base transform for grasping
    def fcn(obj_name):
        pick_unif.problem_env.reset_to_init_state()
        obj = pick_unif.problem_env.env.GetKinBody(obj_name)

        while True:
            print "Calling gengrasp"
            action = pick_unif.predict(obj)
            pick_base_pose, grasp = action
            pick_unif.problem_env.reset_to_init_state()
            print grasp,pick_base_pose
            yield [grasp, pick_base_pose]
    return fcn


def check_traj_collision(problem):
    def fcn(holding_obj_name, grasp, pick_base_conf, obj_name, obj_pose, q_goal, traj):
        return True

        problem.reset_to_init_state()
        obj = problem.env.GetKinBody(obj_name)
        #obj.Enable(True)
        xytheta = obj_pose
        quat = quat_from_z_rot(xytheta[-1])
        set_point(obj, [xytheta[0], xytheta[1], 0.075])
        set_quat(obj, quat)
        robot = problem.robot

        if holding_obj_name == obj_name:
            return False
        holding_obj = problem.env.GetKinBody(holding_obj_name)
        try:
            print holding_obj_name,grasp,pick_base_conf
            problem.apply_pick_action((pick_base_conf, grasp), holding_obj)  # how do I ensure that we are in the same state in both openrave and stripstream?
        except:
            import pdb;pdb.set_trace()
        #draw_robot_base_configs(traj, problem.robot, problem.env)
        #remove_drawn_configs('bconf', problem.env)
        for p in traj:
            set_robot_config(p, problem.robot)
            if problem.env.CheckCollision(robot):
                problem.reset_to_init_state()
                return False
        import pdb;pdb.set_trace()
        problem.reset_to_init_state()
        return False
    return fcn


def gen_placement(problem, place_unif):
    # note generate object placement, relative base conf, absolute base conf, and the path from q1 to abs base conf
    def fcn(obj_name, grasp, pick_base_pose):
        # simulate pick
        obj = problem.env.GetKinBody(obj_name)
        problem.apply_pick_action((pick_base_pose, grasp), obj) # how do I ensure that we are in the same state in both openrave and stripstream?
        place_base_pose = place_unif.predict(obj)
        if not problem.is_collision_at_base_pose(place_base_pose, obj):
            #problem.disable_movable_objects()
            path, is_action_feasible = problem.check_reachability(place_base_pose.squeeze())
            """
            if path is not None:
                for p in path:
                    set_robot_config(p, problem.robot)
                    if problem.env.CheckCollision(problem.robot):
                        import pdb; pdb.set_trace()
                draw_robot_base_configs(path, problem.robot, problem.env)
                import pdb;pdb.set_trace()
                remove_drawn_configs('bconf', problem.env)
                #problem.enable_movable_objects()
            """
            if is_action_feasible == "HasSolution":
                # todo return the object pose as well
                #draw_robot_base_configs(path, problem.robot, problem.env)
                #import  pdb;pdb.set_trace()
                #remove_drawn_configs('bconf', problem.env)
                problem.place_object(place_base_pose)
                obj_pose = get_body_xytheta(obj).squeeze()
                problem.reset_to_init_state()
                return (place_base_pose, obj_pose, path)
            else:
                problem.reset_to_init_state()
                return None
        else:
            problem.reset_to_init_state()
            return None
    return fcn


def read_pddl(filename):
    directory = os.path.dirname(os.path.abspath(__file__))
    return read(os.path.join(directory, filename))


def get_problem():
    convbelt = ConveyorBelt()
    problem_config = convbelt.problem
    directory = os.path.dirname(os.path.abspath(__file__))
    domain_pddl = read(os.path.join(directory, 'domain.pddl'))
    stream_pddl = read(os.path.join(directory, 'stream.pddl'))

    pick_sampler = PickUnif(convbelt, problem_config['env'].GetRobots()[0], problem_config['all_region'])
    place_sampler = PlaceUnif(problem_config['env'], problem_config['env'].GetRobots()[0],
                              problem_config['loading_region'],
                              problem_config['all_region'])
    constant_map = {}
    stream_map = {'gen-grasp': from_gen_fn(gen_grasp(pick_sampler)),
                  'gen-placement': from_fn(gen_placement(convbelt, place_sampler)),
                  #'TrajPoseCollision': fn_from_constant(False)
                  'TrajPoseCollision': from_test(check_traj_collision(convbelt)),
                  }
    obj_names = problem_config['obj_poses'].keys()[0:2]
    obj_poses = problem_config['obj_poses'].values()[0:2]
    init = [('Pickable', obj_name) for obj_name in obj_names]
    #init += [('AtPose', obj_name, obj_pose) for obj_name, obj_pose in zip(obj_names, obj_poses)]
    init += [('Robot', 'pr2')]
    #init += [('Pose', obj_name, obj_pose) for obj_name, obj_pose in zip(obj_names, obj_poses)]
    init += [('BaseConf', [0, 1.05, 0])]
    init += [('EmptyArm',)]

    #goal = ('Picked', 'obj0')
    #goal = ('AtPose', 'obj0', obj_pose)
    goal = ['and', ('InLoadingRegion', 'obj0'),  ('InLoadingRegion', 'obj1')]

    #goal = ('InLoadingRegion', 'obj0')
    convbelt.env.SetViewer('qtcoin')
    return (domain_pddl, constant_map, stream_pddl, stream_map, init, goal), convbelt

##################################################

def process_plan(plan, convbelt):
    convbelt.env.SetViewer('qtcoin')
    convbelt.reset_to_init_state()
    for step_idx, step in enumerate(plan):
        # todo finish this visualization script
        if step[0] == 'pickup':
            obj_name = step[1][0]
            grasp = step[1][1]
            pick_base_pose = step[1][2]
            convbelt.apply_pick_action((pick_base_pose, grasp), convbelt.env.GetKinBody(obj_name))
        else:
            place_obj_name = step[1][0]
            place_base_pose = step[1][4]
            path = step[1][5]
            draw_robot_base_configs(path, convbelt.robot, convbelt.env)
            import pdb;pdb.set_trace()
            remove_drawn_configs('bconf', convbelt.env)
            convbelt.place_object(np.array(place_base_pose))


def view_solution(plan, convbelt):
    convbelt.env.SetViewer('qtcoin')
    robot = convbelt.robot
    convbelt.reset_to_init_state()
    obj_name = plan[0][1][0]
    grasp = plan[0][1][1]
    pick_base_pose = plan[0][1][2]
    import pdb;pdb.set_trace()
    convbelt.apply_pick_action((pick_base_pose, grasp),  convbelt.env.GetKinBody(obj_name))
    place = plan[1][1][3]
    convbelt.place_object(np.array(place))
    import pdb;pdb.set_trace()

    grasp = plan[5][1][1]
    pick_base_pose = plan[5][1][2]
    obj_name = plan[5][1][0]
    obj = convbelt.env.GetKinBody(obj_name)
    convbelt.apply_pick_action((pick_base_pose, grasp), obj)
    import pdb;pdb.set_trace()


def solve_pddlstream():
    pddlstream_problem, convbelt = get_problem()
    solution = solve_incremental(pddlstream_problem, unit_costs=True, max_time=np.inf)
    #solution = solve_focused(pddlstream_problem, unit_costs=True)
    print_solution(solution)
    plan, cost, evaluations = solution
    import pdb;pdb.set_trace()
    process_plan(plan, convbelt)
    import pdb;pdb.set_trace()

    return plan

##################################################


