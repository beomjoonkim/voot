from __future__ import print_function

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../mover_library/')
from samplers import randomly_place_in_region

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

import sys

# obj definitions
min_height = 0.4
max_height = 1

min_width = 0.2
max_width = 0.6

min_length = 0.2
max_length = 0.6


def create_obstacles(env, loading_regions):
    NUM_OBSTACLES = 4
    OBSTACLES = []
    obstacle_poses = {}
    obstacle_shapes = {}
    i = 0
    while len(OBSTACLES) < NUM_OBSTACLES:
        width = np.random.rand(1) * (max_width - min_width) + min_width
        length = np.random.rand(1) * (max_length - min_length) + min_length
        height = np.random.rand(1) * (max_height - min_height) + min_height
        new_body = box_body(env, width, length, height, \
                            name='obst%s' % len(OBSTACLES), \
                            color=(0, (i + .5) / NUM_OBSTACLES, 1))
        trans = np.eye(4);
        trans[2, -1] = 0.075
        env.Add(new_body);
        new_body.SetTransform(trans)
        xytheta = randomly_place_in_region(env, new_body, \
                                           loading_regions[np.random.randint(len(loading_regions))])

        if not (xytheta is None):
            obstacle_shapes['obst%s' % len(OBSTACLES)] = [width[0], length[0], height[0]]
            obstacle_poses['obst%s' % len(OBSTACLES)] = xytheta
            OBSTACLES.append(new_body)
        else:
            env.Remove(new_body)
    return OBSTACLES, obstacle_shapes, obstacle_poses


def create_objects(env, conveyor_belt):
    NUM_OBJECTS = 5
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
        new_body.SetTransform(trans)
        xytheta = randomly_place_in_region(env, new_body, conveyor_belt, np.array([0]))
        OBJECTS.append(new_body)
        obj_shapes['obj%s' % i] = [width[0], length[0], height[0]]
        obj_poses['obj%s' % i] = xytheta
    return OBJECTS, obj_shapes, obj_poses


def load_objects(env, obj_shapes, obj_poses, color):
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


def two_tables_through_door(env, obj_shapes=None, obj_poses=None,
                            obst_shapes=None, obst_poses=None):
    fdir=os.path.dirname(os.path.abspath(__file__))
    env.Load(fdir + '/env.xml')
    robot = env.GetRobots()[0]
    set_default_robot_config(robot)
    region = create_region(env, 'goal', ((-1, 1), (-.3, .3)), \
                           'floorwalls', color=np.array((0, 0, 1, .25)))

    set_config(robot, FOLDED_LEFT_ARM, robot.GetManipulator('leftarm').GetArmIndices())
    set_config(robot, mirror_arm_config(FOLDED_LEFT_ARM), \
               robot.GetManipulator('rightarm').GetArmIndices())

    # left arm IK
    robot.SetActiveManipulator('leftarm')
    manip = robot.GetActiveManipulator()
    ee = manip.GetEndEffector()
    ikmodel1 = databases.inversekinematics.InverseKinematicsModel(robot=robot, \
                                                                  iktype=IkParameterization.Type.Transform6D, \
                                                                  forceikfast=True, freeindices=None, \
                                                                  freejoints=None, manip=None)
    if not ikmodel1.load():
        ikmodel1.autogenerate()

    # right arm torso IK
    robot.SetActiveManipulator('rightarm_torso')
    manip = robot.GetActiveManipulator()
    ee = manip.GetEndEffector()
    ikmodel2 = databases.inversekinematics.InverseKinematicsModel(robot=robot, \
                                                                  iktype=IkParameterization.Type.Transform6D, \
                                                                  forceikfast=True, freeindices=None, \
                                                                  freejoints=None, manip=None)
    if not ikmodel2.load():
        ikmodel2.autogenerate()

    # loading areas
    init_loading_region = AARegion('init_loading_area', \
                                   ((-2.51, -0.81), (-2.51, 0)), \
                                   z=0.0001, color=np.array((1, 0, 1, 0.25)))
    init_loading_region.draw(env)
    init_loading_region2 = AARegion('init_loading_area2', \
                                    ((-2.51, -0.81), (1.7, 2.6)), \
                                    z=0.0001, color=np.array((1, 0, 1, 0.25)))
    init_loading_region2.draw(env)
    init_loading_region4 = AARegion('init_loading_area4', \
                                    ((-2.51, -1.5), (-0.1, 2)), \
                                    z=0.0001, color=np.array((1, 0, 1, 0.25)))
    init_loading_region4.draw(env)
    loading_regions = [init_loading_region, init_loading_region2, \
                       init_loading_region4]

    loading_region = AARegion('loading_area', \
                              ((-2.51, -0.81), (-2.51, 2.51)), \
                              z=0.0001, color=np.array((1, 1, 0, 0.25)))
    loading_region.draw(env)

    # converyor belt region
    conv_x = 2
    conv_y = 1
    conveyor_belt = AARegion('conveyor_belt', \
                             ((-1 + conv_x, 10 * max_width + conv_x), \
                              (-0.4 + conv_y, 0.5 + conv_y)), \
                             z=0.0001, color=np.array((1, 0, 0, 0.25)))

    all_region = AARegion('all_region', \
                          ((-2.51, 10 * max_width + conv_x), (-2.51, 2.51)), \
                          z=0.0001, color=np.array((1, 1, 0, 0.25)))

    if obj_shapes == None:
        OBJECTS, obj_shapes, obj_poses = create_objects(env, conveyor_belt)
    else:
        OBJECTS = load_objects(env, obj_shapes, obj_poses, color=(0, 1, 0))

    if obst_shapes == None:
        OBSTACLES, obst_shapes, obst_poses = create_obstacles(env, loading_regions)
    else:
        OBSTACLES = load_objects(env, obst_shapes, obst_poses, color=(0, 0, 1))

    initial_saver = DynamicEnvironmentStateSaver(env)
    initial_state = (initial_saver, [])
    init_base_conf = np.array([0, 1.05, 0])

    problem = {'initial_state': initial_state,
               'obstacles': OBSTACLES,
               'objects': OBJECTS,
               'loading_region': loading_region,
               'env': env,
               'obst_shapes': obst_shapes,
               'obst_poses': obst_poses,
               'obj_shapes': obj_shapes,
               'obj_poses': obj_poses,
               'all_region': all_region,
               'init_base_conf': init_base_conf}
    return problem  # the second is for indicating 0 placed objs


"""
env=Environment()
problem = two_tables_through_door(env)
"""
