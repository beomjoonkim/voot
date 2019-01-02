from manipulation.problems.fixed import ENVIRONMENTS_DIR
from manipulation.bodies.bodies import box_body, place_xyz_body
from manipulation.problems.problem import *
from manipulation.bodies.bodies import get_name
from misc.functions import randomize
from misc.generators import take
from misc.numerical import INF
from manipulation.bodies.robot import set_default_robot_config
from manipulation.primitives.transforms import get_point, set_point, pose_from_quat_point, unit_quat
from misc.colors import get_color
from manipulation.constants import BODY_PLACEMENT_Z_OFFSET
from manipulation.primitives.utils import Pose

##TODO: Clean this
from manipulation.constants import PARALLEL_LEFT_ARM, REST_LEFT_ARM, HOLDING_LEFT_ARM, FOLDED_LEFT_ARM, \
    FAR_HOLDING_LEFT_ARM, LOWER_TOP_HOLDING_LEFT_ARM, REGION_Z_OFFSET
from manipulation.regions import create_region, AARegion
from manipulation.bodies.bodies import randomly_place_region, place_body, place_body_on_floor
from manipulation.inverse_reachability.inverse_reachability import ir_base_trans
from manipulation.primitives.utils import mirror_arm_config
from manipulation.primitives.transforms import trans_from_base_values, set_pose, set_quat, \
    point_from_pose, axis_angle_from_rot, rot_from_quat, quat_from_pose, quat_from_z_rot, \
    get_pose, base_values_from_pose, pose_from_base_values, set_xy
from itertools import product
import numpy as np
import copy
import math

from manipulation.bodies.bounding_volumes import aabb_extrema, aabb_from_body, aabb_union
from manipulation.inverse_reachability.inverse_reachability import get_custom_ir, get_base_generator
from manipulation.bodies.robot import manip_from_pose_grasp
from manipulation.bodies.robot import get_active_arm_indices
from manipulation.grasps.grasps import FILENAME as GRASP_FILENAME, load_grasp_database
from manipulation.grasps.grasp_options import positive_hash, get_grasp_options
from manipulation.constants import GRASP_APPROACHES, GRASP_TYPES

from manipulation.bodies.bodies import geometry_hash
from manipulation.bodies.bounding_volumes import aabb_from_body
from manipulation.grasps.grasps import save_grasp_database, Grasp
from openravepy import *
import socket

import sys

sys.path.append('../mover_library/')
from samplers import *
from utils import *
from operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp
from misc.priority_queue import Stack, Queue, PriorityQueue
from TreeNode import *
import pickle
# obj definitions
min_height = 0.4
max_height = 1

min_width = 0.2
max_width = 0.6

min_length = 0.2
max_length = 0.6

SLEEPTIME = 0.05


def compute_fetch_vec(key_configs, fetch_path, robot, env):
    fetch_vec = [[]] * len(key_configs)
    xy_threshold = 0.3  # size of the base - 0.16
    th_threshold = 20 * np.pi / 180  # adhoc
    fetch_path = fetch_path[::int(0.1 * len(fetch_path))]
    for f in fetch_path:
        for kidx, k in enumerate(key_configs):
            xy_dist = np.linalg.norm(f[0:2] - k[0:2])
            th_dist = abs(f[2] - k[2]) if abs(f[2] - k[2]) < np.pi else 2 * np.pi - abs(f[2] - k[2])
            if xy_dist < xy_threshold and th_dist < th_threshold:
                fetch_vec[kidx] = 1
            else:
                fetch_vec[kidx] = 0
    return fetch_vec


def convert_collision_vec_to_one_hot(c_data):
    n_konf = c_data.shape[1]
    onehot_cdata = []
    for cvec in c_data:
        one_hot_cvec = np.zeros((n_konf, 2))
        for boolean_collision, onehot_collision in zip(cvec, one_hot_cvec):
            onehot_collision[boolean_collision] = 1
        assert (np.all(np.sum(one_hot_cvec, axis=1) == 1))
        onehot_cdata.append(one_hot_cvec)

    onehot_cdata = np.array(onehot_cdata)
    return onehot_cdata


def draw_robot_at_conf(conf, transparency, name, robot, env):
    newrobot = RaveCreateRobot(env, robot.GetXMLId())
    newrobot.Clone(robot, 0)
    newrobot.SetName(name)
    env.Add(newrobot, True)
    set_robot_config(conf, newrobot)
    for link in newrobot.GetLinks():
        for geom in link.GetGeometries():
            geom.SetTransparency(transparency)


def draw_robot_at_path(path, robot, env):
    for conf in path:
        draw_robot_at_conf(conf, 0.7, 'path', robot, env)


def pick_obj(obj, robot, g_configs, left_manip, right_manip):
    set_config(robot, g_configs[0], left_manip.GetArmIndices())
    set_config(robot, g_configs[1], right_manip.GetArmIndices())
    sleep(SLEEPTIME)
    robot.Grab(obj)


def place_obj(obj, robot, leftarm_manip, rightarm_manip):
    sleep(SLEEPTIME)
    robot.Release(obj)
    set_config(robot, FOLDED_LEFT_ARM, leftarm_manip.GetArmIndices())
    set_config(robot, mirror_arm_config(FOLDED_LEFT_ARM), \
               rightarm_manip.GetArmIndices())


"""
def compute_obj_collisions(env,robot,path,objs):
  obj_names = []
  # set robot to its initial configuration
  with robot:
    for p in path:
      set_robot_config(p,robot)
      for obj in objs:
        if env.CheckCollision(robot,obj) and not (obj.GetName() in obj_names):
          obj_names.append(obj.GetName())
  return obj_names
"""


def create_objects(env, obj_region, table_region):
    NUM_OBJECTS = 10
    OBJECTS = []
    obj_shapes = {}
    obj_poses = {}
    for i in range(NUM_OBJECTS):
        width = np.random.rand(1) * (max_width - min_width) + min_width
        length = np.random.rand(1) * (max_width - min_length) + min_length
        height = np.random.rand(1) * (max_height - min_height) + min_height
        new_body = box_body(env, width, length, height, \
                            name='obj%s' % i, \
                            color=(0, (i + .5) / NUM_OBJECTS, 0))
        trans = np.eye(4);
        trans[2, -1] = 0.075
        env.Add(new_body);
        new_body.SetTransform(trans);
        if i == 0:
            xytheta = randomly_place_in_region(env, new_body, table_region)
        else:
            xytheta = randomly_place_in_region(env, new_body, obj_region)
        OBJECTS.append(new_body)
        obj_shapes['obj%s' % i] = [width[0], length[0], height[0]]
        obj_poses['obj%s' % i] = xytheta
    return OBJECTS, obj_poses, obj_shapes


def load_objects(env, obj_shapes, obj_poses, color):
    # sets up the object at their locations in the original env
    OBJECTS = []
    i = 0
    nobj = len(obj_shapes.keys())
    for obj_name in obj_shapes.keys():
        xytheta = obj_poses[obj_name]
        width, length, height = obj_shapes[obj_name]
        quat = quat_from_z_rot(xytheta[-1])

        new_body = box_body(env, width, length, height, \
                            name=obj_name, \
                            color=np.array(color) / float(nobj - i))
        i += 1
        env.Add(new_body);
        set_point(new_body, [xytheta[0], xytheta[1], 0.075])
        set_quat(new_body, quat)
        OBJECTS.append(new_body)
    return OBJECTS


def sample_pick(obj, robot, env, region, n_trials=5):
    # takes the target object to be picked and returns:
    # - base pose to pick the object
    # - grasp parameters to pick the object
    # - grasp to pick the object
    # to which there is a valid path

    # n_trials = 5 # try 20 different samples
    original_trans = robot.GetTransform()
    n_motion_plan_limit = 2
    n_motion_plans = 0
    for _ in range(n_trials):
        # sample base pose
        base_pose = sample_ir(obj, robot, env, region)
        if base_pose is None:
            continue
        # base pose is the world x,y,th_z of the robot;
        # its validity is checked by set_robot_confg (see sample_ir, line 132)
        set_robot_config(base_pose, robot)

        # sample grasp
        theta, height_portion, depth_portion = sample_grasp_parameters()
        grasps = compute_two_arm_grasp(depth_portion, height_portion, theta, obj, robot)
        g_config = solveTwoArmIKs(env, robot, obj, grasps)
        if g_config is None:
            continue

        # check reachability
        robot.SetTransform(original_trans)
        # print 'pick motion planning...'
        path, tpath, status = get_motion_plan(robot, base_pose, env, n_node_lim=np.inf, maxiter=20)
        print 'done', status, tpath
        if status == 'HasSolution':
            # found feasible pick parameter
            return base_pose, [theta, height_portion, depth_portion], g_config, path

        n_motion_plans += 1
        if n_motion_plans == n_motion_plan_limit:
            print "pick reached n motion plan limit"
            break

    robot.SetTransform(original_trans)
    return None, None, None, None


def sample_placement(env, robot, region, prev_base_pose=None):
    status = "Failed"
    path = None
    original_trans = robot.GetTransform()
    tried = []
    n_trials = 5  # try 20 different samples
    n_motion_plan_limit = 2
    n_motion_plans = 0
    for _ in range(n_trials):
        # this works for this domain because the robot and object region are the same
        xytheta = randomly_place_in_region(env, robot, region)
        if prev_base_pose is not None:
            while np.linalg.norm(np.array(xytheta) - np.array(prev_base_pose)) < 0.01:
                xytheta = randomly_place_in_region(env, robot, region)
        if xytheta is None:
            continue

        # this assertion snippet verifies xytheta is indeed world x,y, and th_z
        # using set_robot_config
        target_T = robot.GetTransform()
        set_robot_config(xytheta, robot)
        verify_T = robot.GetTransform()
        assert (np.all(np.isclose(target_T, verify_T)))
        robot.SetTransform(original_trans)
        path, tpath, status = get_motion_plan(robot, xytheta, env, n_node_lim=np.inf, maxiter=5)
        print 'done', status, tpath
        if status == "HasSolution":
            return xytheta, path

        n_motion_plans += 1
        if n_motion_plans == n_motion_plan_limit:
            print "reached n motion plan limit"
            break
    robot.SetTransform(original_trans)
    return None, None


def sample_placement_using_soap(env, robot, region, Gplace, node, fetch_vec):
    status = "Failed"
    path = None
    original_trans = robot.GetTransform()
    tried = []
    n_trials = 5  # try 20 different samples

    key_configs = Gplace.key_configs
    n_key_confs = len(key_configs)

    # make c
    try:
        c_data = node.c_data
        print 'saved cdata time'
    except:
        stime = time.time()
        c_data = compute_occ_vec(key_configs, robot, env)[None, :] * 1
        print 'occ time', time.time() - stime
        c_data = convert_collision_vec_to_one_hot(c_data)
        c_data = np.concatenate([c_data, fetch_vec], axis=-1)
        node.c_data = c_data

    # make w
    tobj = env.GetKinBody('obj0')
    o_data = get_body_xytheta(tobj)
    c0_data = get_robot_xytheta(robot)
    clean_pose_data(c0_data)
    scaled_c0_data = Gplace.c0_scaler.transform(c0_data)
    w_data = np.hstack([scaled_c0_data, o_data])

    # make k
    n_gen = 100
    zvals = np.random.normal(size=(n_gen, Gplace.dim_z)).astype('float32')
    w_data = np.tile(w_data, (n_gen, 1))
    c_data = np.tile(c_data, (n_gen, 1, 1))

    stime = time.time()
    Gpred = Gplace.a_gen.predict([zvals, w_data, c_data])
    Gpred = Gplace.x_scaler.inverse_transform(Gpred)
    p_samples = Gpred

    print 'place prediction time', time.time() - stime

    VISUALIZE = False
    if VISUALIZE == True:
        draw_configs(configs=p_samples, env=env, name='conf', colors=(1, 0, 0), transparency=0)
        if env.GetViewer() is None:
            env.SetViewer('qtcoin')
        remove_drawn_configs('conf', env)

    max_path_plan_tries = 3
    n_tries = 0
    obj = robot.GetGrabbed()[0]
    sample_idxs = range(100)
    np.random.shuffle(sample_idxs)
    for idx in sample_idxs:
        eps = np.random.rand()
        if eps > 0.01:
            xytheta = p_samples[idx]
        else:
            print 'using random sample'
            xytheta = randomly_place_in_region(env, robot, region)
        set_robot_config(xytheta, robot)
        inCollision = (check_collision_except(obj, robot, env)) \
                      or (check_collision_except(robot, obj, env))
        inRegion = (region.contains(robot.ComputeAABB())) and \
                   (region.contains(obj.ComputeAABB()))
        if inCollision or not inRegion: continue

        robot.SetTransform(original_trans)
        print "path planning..."
        # path,tpath,status = get_motion_plan(robot,xytheta,env,maxiter=10,n_node_lim=10000)
        path, tpath, status = get_motion_plan(robot, xytheta, env, n_node_lim=10000, maxiter=20)
        print 'path planning took ', tpath
        if status == "HasSolution":
            print "returning with soln"
            return xytheta, path
        n_tries += 1
        if n_tries > max_path_plan_tries:
            break
    xytheta = None
    path = None

    # if we never tried path planning, then call regular planner
    # print 'all tries failed, relying on regular planner'
    # robot.SetTransform(original_trans)
    # if n_tries==0:
    #  xytheta,path = sample_placement( env,robot,region)
    robot.SetTransform(original_trans)
    if xytheta is None:
        print 'regular place failed'

    robot.SetTransform(original_trans)
    return xytheta, path


def sample_pick_using_soap(obj, obj_shape, robot, env, region, Gpick, node, fetch_vec):
    original_trans = robot.GetTransform()

    # make w
    c0_data = get_robot_xytheta(robot)
    clean_pose_data(c0_data)
    scaled_c0_data = Gpick.c0_scaler.transform(c0_data)

    opose_data = get_body_xytheta(obj)
    clean_pose_data(opose_data)
    scaled_opose_data = Gpick.opose_scaler.transform(opose_data)

    oshape_data = np.array(obj_shape)[None, :]
    scaled_oshape_data = Gpick.oshape_scaler.transform(oshape_data)

    w_data = np.hstack([scaled_c0_data, scaled_opose_data, scaled_oshape_data])

    key_configs = Gpick.key_configs

    # make c
    try:
        c_data = node.c_data
        print 'saved cdata time'
    except:
        stime = time.time()
        c_data = compute_occ_vec(key_configs, robot, env)[None, :] * 1
        print 'occ time', time.time() - stime
        c_data = convert_collision_vec_to_one_hot(c_data)
        c_data = np.concatenate([c_data, fetch_vec], axis=-1)
        node.c_data = c_data

    # generate values
    n_gen = 90
    zvals = np.random.normal(size=(n_gen, Gpick.dim_z)).astype('float32')
    w_data = np.tile(w_data, (n_gen, 1))
    c_data = np.tile(c_data, (n_gen, 1, 1))

    stime = time.time()
    Gpred = Gpick.a_gen.predict([zvals, w_data, c_data])
    Gpred = Gpick.x_scaler.inverse_transform(Gpred)
    p_samples = Gpred
    np.random.shuffle(p_samples)
    print 'pick prediction time', time.time() - stime

    # get the predicted constraints
    base_poses = copy.deepcopy(p_samples[:, -2:])  # + (np.random.rand(90,2)*0.2-0.1)
    base_poses[:, -2:] = copy.deepcopy(base_poses[:, -2:] + opose_data[0, 0:2])
    grasps = p_samples[:, :3]
    base_poses = np.hstack([base_poses, np.zeros((base_poses.shape[0], 1))])

    base_poses_ = base_poses

    # Visualization for debugging
    VISUALIZE = False
    if VISUALIZE == True:
        for base_pose in base_poses:
            angle = sample_angle_facing_given_transform(robot, opose_data[0, 0:2], base_pose[0:2])
            base_pose[-1] = angle
        base_poses_to_visualize = []
        with robot:
            for b in base_poses:
                set_robot_config(b, robot)
                if env.CheckCollision(robot):
                    continue
                base_poses_to_visualize.append(b)

        draw_configs(configs=base_poses_to_visualize, env=env, name='conf', colors=(1, 0, 0), transparency=0.5)

        with robot:
            uniform_samples = [sample_ir(obj, robot, env, region) for i in range(100)]
        draw_configs(configs=uniform_samples, env=env, name='unif_conf', colors=(0, 0, 1), transparency=0.5)
        if env.GetViewer() is None:
            env.SetViewer('qtcoin')
        remove_drawn_configs('conf', env)
        remove_drawn_configs('unif_conf', env)

    base_poses = base_poses_
    max_path_plan_tries = 1
    n_tries = 0

    for base_pose, grasp in zip(base_poses, grasps):
        robot.SetTransform(original_trans)
        choose_p = np.random.random()
        if choose_p < 0.01:
            print 'chose to do uniform sampling'
            bp, gp, gc, p = sample_pick(obj, robot, env, region, n_trials=1)
            if bp is not None:
                print 'uniform sampling worked out'
                return bp, gp, gc, p
        # get base pose
        g_config = None
        for _ in range(5):
            angle = sample_angle_facing_given_transform(robot, opose_data[0, 0:2], base_pose[0:2])
            base_pose[-1] = angle
            set_robot_config(base_pose, robot)
            if env.CheckCollision(robot):
                continue

            # get grasp parameters
            theta = grasp[0];
            height_portion = grasp[1];
            depth_portion = grasp[2]

            # compute IK and grasps
            grasps = compute_two_arm_grasp(depth_portion, height_portion, theta, obj, robot)
            g_config = solveTwoArmIKs(env, robot, obj, grasps)
            if g_config is not None:
                print 'grasp found'
                break

        # have we found a g_config?
        if g_config is None:
            continue

        # if we have, then try to plan a path
        print "path planning..."
        robot.SetTransform(original_trans)
        # path,tpath,status = get_motion_plan(robot,base_pose,env,maxiter=10,n_node_lim=10000)
        path, tpath, status = get_motion_plan(robot, base_pose, env, n_node_lim=10000, maxiter=20)
        print 'path planning took ', tpath
        if status == 'HasSolution':
            print "returning with a solution"
            return base_pose, [theta, height_portion, depth_portion], g_config, path

        n_tries += 1
        if n_tries > max_path_plan_tries:
            break

    base_pose = None
    grasp_params = None
    g_config = None
    path = None

    robot.SetTransform(original_trans)
    return base_pose, grasp_params, g_config, path


def compute_fetch_path(target_obj, target_obj_shape, objects, robot, env, all_region):
    path = None
    try_limit = 10
    n_try = 0
    while path is None:
        for obj in objects:
            if obj is target_obj: continue
            obj.Enable(False)

        print "Sampling grasp"

        # if Gpick is None:
        base_pose, grasp_params, g_config, path = sample_pick(target_obj, robot, env, all_region)
        """
        else: 
          base_pose,grasp_params,g_config,path = sample_pick_using_soap(target_obj,\
                                                   target_obj_shape,robot,env,all_region,Gpick)
        """
        n_try += 1
        if n_try > try_limit:
            return None, None, None, None
    for obj in objects:
        obj.Enable(True)
    return base_pose, grasp_params, g_config, path


def NAMO_problem(env, pfile):
    is_new_problem_instance = not os.path.isfile(pfile)
    if not is_new_problem_instance:
        problem_config = pickle.load(open(pfile, 'r'))
        obj_shapes = problem_config['obj_shapes']
        obj_poses = problem_config['obj_poses']
        path = problem_config['original_path']
        try:
            collided_obj_names = problem_config['collided_obj_names']
        except KeyError:
            collided_obj_names = problem_config['collided_objs']
    env.Load('./namo/env.xml')
    robot = env.GetRobots()[0]
    set_point(env.GetKinBody('shelf1'), [1, -2.33205483, 0.010004])
    set_point(env.GetKinBody('shelf2'), [1, 2.33205483, 0.010004])

    robot_initial_pose = get_pose(robot)
    leftarm_manip = robot.GetManipulator('leftarm')
    rightarm_manip = robot.GetManipulator('rightarm')
    rightarm_torso_manip = robot.GetManipulator('rightarm_torso')

    set_config(robot, FOLDED_LEFT_ARM, robot.GetManipulator('leftarm').GetArmIndices())
    set_config(robot, mirror_arm_config(FOLDED_LEFT_ARM),
               robot.GetManipulator('rightarm').GetArmIndices())

    robot_initial_config = np.array([-1, 1, 0])
    set_robot_config(robot_initial_config, robot)

    # left arm IK
    robot.SetActiveManipulator('leftarm')
    manip = robot.GetActiveManipulator()
    ee = manip.GetEndEffector()
    ikmodel1 = databases.inversekinematics.InverseKinematicsModel(robot=robot,
                                                                  iktype=IkParameterization.Type.Transform6D,
                                                                  forceikfast=True, freeindices=None,
                                                                  freejoints=None, manip=None)
    if not ikmodel1.load():
        ikmodel1.autogenerate()

    # right arm torso IK
    robot.SetActiveManipulator('rightarm_torso')
    manip = robot.GetActiveManipulator()
    ee = manip.GetEndEffector()
    ikmodel2 = databases.inversekinematics.InverseKinematicsModel(robot=robot,
                                                                  iktype=IkParameterization.Type.Transform6D,
                                                                  forceikfast=True, freeindices=None,
                                                                  freejoints=None, manip=None)
    if not ikmodel2.load():
        ikmodel2.autogenerate()

    region = create_region(env, 'goal', ((-1, 1), (-.3, .3)),
                           'floorwalls', color=np.array((0, 0, 1, .25)))
    obj_region = region = AARegion('obj_region', ((-1.51 * 0.3, 2.51), (-2.51, 1.5)),
                                   z=0.0001, color=np.array((1, 1, 0, 0.25)))
    table_region = region = AARegion('table_region', ((-2.51 * 0.1, 2.51), (-2.51, -1)),
                                     z=0.0001, color=np.array((1, 0, 1, 0.25)))
    all_region = region = AARegion('all_region', ((-2.51, 2.51), (-2.51, 2.51)),
                                   z=0.0001, color=np.array((1, 1, 0, 0.25)))
    # obj_region.draw(env)
    # table_region.draw(env)
    if is_new_problem_instance:
        OBJECTS, obj_poses, obj_shapes = create_objects(env, obj_region, table_region)
    else:
        OBJECTS = load_objects(env, obj_shapes, obj_poses, (0, 1, 0))

    # compute swept volume
    target_obj = env.GetKinBody('obj0')
    set_color(target_obj, (1, 0, 0))

    target_obj_shape = obj_shapes[target_obj.GetName()]
    problem_hard = False

    if is_new_problem_instance:
        while not problem_hard:
            set_robot_config(robot_initial_config, robot)
            base_pose, grasp_params, g_config, path = compute_fetch_path(target_obj, target_obj_shape, \
                                                                         OBJECTS, robot, env, all_region)
            if path is None:
                print "This problem is infeasible"
                env.Destroy()
                RaveDestroy
                return None
            set_robot_config(robot_initial_config, robot)
            collided_obj_names = compute_obj_collisions(env, robot, path, OBJECTS)

            problem_hard = len(collided_obj_names) >= 5
            print len(collided_obj_names)
            if not problem_hard:
                for obj in OBJECTS: env.Remove(obj)
                OBJECTS, obj_poses, obj_shapes = create_objects(env, obj_region, table_region)
                target_obj = OBJECTS[0]
                target_obj_shape = obj_shapes[target_obj.GetName()]
                print "Problem not hard"

    initial_saver = DynamicEnvironmentStateSaver(env)
    initial_state = (initial_saver, collided_obj_names)

    if is_new_problem_instance:
        pickle.dump({'obj_shapes': obj_shapes, 'obj_poses': obj_poses,
                     'collided_obj_names': collided_obj_names, 'grasp_params': grasp_params,
                     'g_config': g_config, 'base_pose': base_pose, 'original_path': path},
                    open(pfile, 'wb'))

    problem = {'initial_state': initial_state,
               'robot_initial_config': robot_initial_config,
               'objects': OBJECTS,
               'obj_region': obj_region,
               'table_region': table_region,
               'all_region': all_region,
               'env': env,
               'obj_shapes': obj_shapes,
               'collided_objs': collided_obj_names,
               'obj_poses': obj_poses,
               'target_obj': target_obj,
               'original_path': path}

    # order objects according to MCR?
    return problem
