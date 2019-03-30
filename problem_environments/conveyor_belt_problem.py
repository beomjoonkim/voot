from __future__ import print_function

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../mover_library/')

from manipulation.bodies.bodies import box_body
from manipulation.problems.problem import *
from manipulation.bodies.robot import set_default_robot_config
from manipulation.primitives.transforms import get_point, set_point, pose_from_quat_point, unit_quat
from manipulation.constants import *

##TODO: Clean this
from manipulation.constants import PARALLEL_LEFT_ARM, REST_LEFT_ARM, HOLDING_LEFT_ARM, FOLDED_LEFT_ARM, \
    FAR_HOLDING_LEFT_ARM, LOWER_TOP_HOLDING_LEFT_ARM, REGION_Z_OFFSET
from manipulation.regions import create_region, AARegion
from manipulation.primitives.utils import mirror_arm_config
from manipulation.primitives.transforms import trans_from_base_values, set_pose, set_quat, \
    point_from_pose, axis_angle_from_rot, rot_from_quat, quat_from_pose, quat_from_z_rot, \
    get_pose, base_values_from_pose, pose_from_base_values, set_xy, quat_from_angle_vector, \
    quat_from_trans

from manipulation.primitives.savers import DynamicEnvironmentStateSaver

import numpy as np


from manipulation.bodies.bodies import set_config

# search episode

from manipulation.primitives.inverse_kinematics import *
from manipulation.motion.trajectories import *
from manipulation.constants import *
from mover_library.samplers import randomly_place_in_region
from mover_library.utils import *

import pickle
# obj definitions
min_height = 0.4
max_height = 1

min_width = 0.2
max_width = 0.6
min_length = 0.2
max_length = 0.6


def create_obstacles(env, loading_regions):
    num_obstacles = 3
    obstacles = []
    obstacle_poses = {}
    obstacle_shapes = {}
    i = 0
    min_width = 0.2
    max_width = 0.4
    min_length = 0.2
    max_length = 3
    max_height = 0.5
    while len(obstacles) < num_obstacles:
        if len(obstacles) == 0:
            min_length = 1.8
            max_length = 1.8
        else:
            min_length = 0.2
            max_length = 0.8


        width = np.random.rand(1) * (max_width - min_width) + min_width
        length = np.random.rand(1) * (max_length - min_length) + min_length
        height = np.random.rand(1) * (max_height - min_height) + min_height
        new_body = box_body(env, width, length, height,
                            name='obst%s' % len(obstacles),
                            color=(0, (i + .5) / num_obstacles, 1))
        trans = np.eye(4);
        trans[2, -1] = 0.075
        env.Add(new_body);
        new_body.SetTransform(trans)
        xytheta= randomly_place_in_region(env, new_body, loading_regions)
                                           #loading_regions[np.random.randint(len(loading_regions))])

        if not (xytheta is None):
            obstacle_shapes['obst%s' % len(obstacles)] = [width[0], length[0], height[0]]
            obstacle_poses['obst%s' % len(obstacles)] = xytheta
            obstacles.append(new_body)
        else:
            raise ValueError, 'Not enough spot for obstacles'

            #env.Remove(new_body)
    return obstacles, obstacle_shapes, obstacle_poses


def create_objects(env, conveyor_belt):
    num_objects = 5
    objects = []
    obj_shapes = {}
    obj_poses = {}

    for i in range(num_objects):
        if i > 10 and i < 15:
            min_width = 0.7
            max_width = 0.7
            min_length = 0.6
        else:
            min_width = 0.2
            max_width = 0.6
            min_length = 0.2

        width = np.random.rand(1) * (max_width - min_width) + min_width
        length = np.random.rand(1) * (max_width - min_length) + min_length
        height = np.random.rand(1) * (max_height - min_height) + min_height
        new_body = box_body(env, width, length, height, \
                            name='obj%s' % i, \
                            color=(0, (i + .5) / num_objects, 0))
        trans = np.eye(4);
        trans[2, -1] = 0.075
        env.Add(new_body);
        new_body.SetTransform(trans)
        xytheta = randomly_place_in_region(env, new_body, conveyor_belt)
        objects.append(new_body)
        obj_shapes['obj%s' % i] = [width[0], length[0], height[0]]
        obj_poses['obj%s' % i] = xytheta
    return objects, obj_shapes, obj_poses


def load_objects(env, obj_shapes, obj_poses, color):
    objects = []
    i = 0
    nobj = len(obj_shapes.keys())
    for obj_name in obj_shapes.keys():
        xytheta = obj_poses[obj_name].squeeze()
        width, length, height = obj_shapes[obj_name]
        quat = quat_from_z_rot(xytheta[-1])

        new_body = box_body(env, width, length, height, \
                            name=obj_name, \
                            color=np.array(color) / float(nobj - i))
        i += 1
        env.Add(new_body);
        set_point(new_body, [xytheta[0], xytheta[1], 0.075])
        set_quat(new_body, quat)
        objects.append(new_body)
    return objects


def create_conveyor_belt_problem(env, obj_setup=None):
    if obj_setup is not None:
        obj_shapes = obj_setup['object_shapes']
        obj_poses = obj_setup['object_poses']
        obst_shapes = obj_setup['obst_shapes']
        obst_poses = obj_setup['obst_poses']

    fdir=os.path.dirname(os.path.abspath(__file__))
    env.Load(fdir + '/convbelt_env_diffcult_shapes.xml')
    env.SetViewer('qtcoin')
    robot = env.GetRobots()[0]
    set_default_robot_config(robot)

    set_config(robot, FOLDED_LEFT_ARM, robot.GetManipulator('leftarm').GetArmIndices())
    set_config(robot, mirror_arm_config(FOLDED_LEFT_ARM),
               robot.GetManipulator('rightarm').GetArmIndices())

    # left arm IK
    robot.SetActiveManipulator('leftarm')
    ikmodel1 = databases.inversekinematics.InverseKinematicsModel(robot=robot,
                                                                  iktype=IkParameterization.Type.Transform6D,
                                                                  forceikfast=True, freeindices=None,
                                                                  freejoints=None, manip=None)
    if not ikmodel1.load():
        ikmodel1.autogenerate()

    # right arm torso IK
    robot.SetActiveManipulator('rightarm_torso')
    ikmodel2 = databases.inversekinematics.InverseKinematicsModel(robot=robot,
                                                                  iktype=IkParameterization.Type.Transform6D,
                                                                  forceikfast=True, freeindices=None,
                                                                  freejoints=None, manip=None)
    if not ikmodel2.load():
        ikmodel2.autogenerate()

    # loading areas
    loading_region = AARegion('loading_area', ((-3.51, -0.81), (-2.51, 2.51)), z=0.01, color=np.array((1, 1, 0, 0.25)))

    # converyor belt region
    conv_x = 3
    conv_y = 1
    conveyor_belt = AARegion('conveyor_belt', ((-1 + conv_x, 20 * max_width + conv_x),
                                               (-0.4 + conv_y, 0.5 + conv_y)), z=0.01, color=np.array((1, 0, 0, 0.25)))

    all_region = AARegion('all_region', ((-3.51, 20 * max_width + conv_x),
                                         (-2.51, 2.51)), z=0.01, color=np.array((1, 1, 0, 0.25)))

    """
    if obj_setup is None:
        objects, obj_shapes, obj_poses = create_objects(env, conveyor_belt)
        obstacles, obst_shapes, obst_poses = create_obstacles(env, loading_region)
    else:
        objects = load_objects(env, obj_shapes, obj_poses, color=(0, 1, 0))
        obstacles = load_objects(env, obst_shapes, obst_poses, color=(0, 0, 1))

    #set_obj_xytheta([-1, -1, 1], obstacles[0])
    #set_obj_xytheta([-2, 2.3, 0], obstacles[1])
    #obst_poses = [randomly_place_in_region(env, obj, loading_region) for obj in obstacles]
    #obst_poses = [get_body_xytheta(obj) for obj in obstacles]

    """
    """
    tobj = env.GetKinBody('tobj3')
    tobj_xytheta = get_body_xytheta(tobj.GetLinks()[1])
    tobj_xytheta[0, -1] = (160 / 180.0) * np.pi
    set_obj_xytheta(tobj_xytheta, tobj.GetLinks()[1])
    """
    init_base_conf = np.array([0, 1.05, 0])
    set_robot_config(np.array([0, 1.05, 0]), robot)
    objects = []
    for tobj in env.GetBodies():
        if tobj.GetName().find('tobj') == -1: continue
        randomly_place_in_region(env, tobj, conveyor_belt)
        objects.append(tobj)

    # todo make infinite sequence of objects

    initial_saver = DynamicEnvironmentStateSaver(env)
    initial_state = (initial_saver, [])
    problem = {'initial_state': initial_state,
               #'obstacles': obstacles,
               'objects': objects,
               'conveyor_belt_region':conveyor_belt,
               'loading_region': loading_region,
               'env': env,
               #'obst_shapes': obst_shapes,
               #'obst_poses': obst_poses,
               #'obj_shapes': obj_shapes,
               #'obj_poses': obj_poses,
               'entire_region': all_region,
               'init_base_conf': init_base_conf}
    return problem  # the second is for indicating 0 placed objs


"""
env=Environment()
problem = two_tables_through_door(env)
"""
