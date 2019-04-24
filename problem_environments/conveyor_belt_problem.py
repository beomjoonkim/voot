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
min_height = 0.7
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


def create_objects(env, conveyor_belt, num_objects):
    objects = []
    obj_shapes = {}
    obj_poses = {}

    for i in range(num_objects):
        #if 0 <= i < 0:
        if i == 0 or i == 1:
            min_width = 0.6
            max_width = 0.7
            min_length = 0.7
            name = 'big_obj'
        else:
            min_width = 0.4
            max_width = 0.5
            min_length = 0.6
            name = 'small_obj'
        width = np.random.rand(1) * (max_width - min_width) + min_width
        length = np.random.rand(1) * (max_width - min_length) + min_length
        height = np.random.rand(1) * (max_height - min_height) + min_height
        new_body = box_body(env, width, length, height,
                            name=name+'%s' % i,
                            color=(0, (i + .5) / num_objects, 0))
        trans = np.eye(4)
        trans[2, -1] = 0.075
        env.Add(new_body)
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

        new_body = box_body(env, width, length, height,
                            name=obj_name,
                            color=np.array(color) / float(nobj - i))
        i += 1
        env.Add(new_body)
        set_point(new_body, [xytheta[0], xytheta[1], 0.075])
        set_quat(new_body, quat)
        objects.append(new_body)
    return objects


def create_conveyor_belt_problem(env, obj_setup=None, problem_idx=0):
    if obj_setup is not None:
        obj_shapes = obj_setup['object_shapes']
        obj_poses = obj_setup['object_poses']
        obst_shapes = obj_setup['obst_shapes']
        obst_poses = obj_setup['obst_poses']

    fdir=os.path.dirname(os.path.abspath(__file__))

    if problem_idx == 0 or problem_idx == 1:
        env.Load(fdir + '/convbelt_env_diffcult_shapes.xml')
    else:
        env.Load(fdir + '/convbelt_env_diffcult_shapes_two_rooms.xml')

    """
    if problem_idx == 0:
        env.Load(fdir + '/convbelt_env_diffcult_shapes.xml')
    else:
        env.Load(fdir + '/convbelt_env.xml')
    """

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
    """
    self.home_region_xy = [x_extents / 2.0, 0]
    self.home_region_xy_extents = [x_extents, y_extents]
    self.home_region = AARegion('entire_region',
                                ((-x_extents + self.home_region_xy[0], x_extents + self.home_region_xy[0]),
                                 (-y_extents, y_extents)), z=0.135, color=np.array((1, 1, 0, 0.25)))
    """
    init_base_conf = np.array([0, 1.05, 0])
    set_robot_config(np.array([0, 1.05, 0]), robot)

    # converyor belt region
    conv_x = 3
    conv_y = 1
    conveyor_belt = AARegion('conveyor_belt', ((-1 + conv_x, 20 * max_width + conv_x),
                                               (-0.4 + conv_y, 0.5 + conv_y)), z=0.01, color=np.array((1, 0, 0, 0.25)))

    y_extents = 5.0
    x_extents = 3.01
    entire_region = AARegion('entire_region', ((-7.4, 20 * max_width + conv_x), (-y_extents - 2.5, y_extents - 2)), z=0.01, color=np.array((1, 1, 0, 0.25)))
    loading_region = AARegion('loading_area', ((-7.4, -0.5), (-7.5, 3.0)), z=0.01, color=np.array((1, 1, 0, 0.25)))

    big_region_1 = AARegion('big_region_1', ((-5, -0.5), (-7.5, -0.4)), z=0.01, color=np.array((1, 1, 0, 0.25)))
    big_region_2 = AARegion('big_region_2', ((-7.4, -4.0), (-7.5, 3.0)), z=0.01, color=np.array((1, 1, 0, 0.25)))

    if problem_idx == 0 or problem_idx == 1:
        objects = []
        i = 1
        for tobj in env.GetBodies():
            if tobj.GetName().find('tobj') == -1: continue
            randomly_place_in_region(env, tobj, conveyor_belt)
            set_obj_xytheta([2 + i, 1.05, 0], tobj)
            objects.append(tobj)
            i += 1.1

        square_objects, obj_shapes, obj_poses = create_objects(env, conveyor_belt, num_objects=10)
        objects += square_objects
    else:
        objects = []
        i = 1
        for tobj in env.GetBodies():
            if tobj.GetName().find('tobj') == -1: continue
            randomly_place_in_region(env, tobj, conveyor_belt)
            set_obj_xytheta([2 + i, 1.05,  get_body_xytheta(tobj)[0, -1]], tobj)
            #objects.append(tobj)
            i += 1.1

        square_objects, obj_shapes, obj_poses = create_objects(env, conveyor_belt, num_objects=10)
        for obj in square_objects:
            set_obj_xytheta([2 + i, 1.05, get_body_xytheta(obj)[0, -1]], obj)
            objects.append(obj)
            i += 1.1

        #objects += square_objects

    """
    if problem_idx == 0:
        objects = []
        i = 1
        for tobj in env.GetBodies():
            if tobj.GetName().find('tobj') == -1: continue
            randomly_place_in_region(env, tobj, conveyor_belt)
            set_obj_xytheta([2+i, 1.05, 0], tobj)
            objects.append(tobj)
            i += 1.1

        square_objects, obj_shapes, obj_poses = create_objects(env, conveyor_belt, num_objects=10)
        objects += square_objects
    else:
        objects, obj_shapes, obj_poses = create_objects(env, conveyor_belt, num_objects=20)
    """

    initial_saver = DynamicEnvironmentStateSaver(env)
    initial_state = (initial_saver, [])
    problem = {'initial_state': initial_state,
               'objects': objects,
               'conveyor_belt_region': conveyor_belt,
               'loading_region': loading_region,
               'big_region_1': big_region_1,
               'big_region_2': big_region_2,
               'env': env,
               'entire_region': entire_region,
               'init_base_conf': init_base_conf}
    return problem  # the second is for indicating 0 placed objs


"""
env=Environment()
problem = two_tables_through_door(env)
"""
