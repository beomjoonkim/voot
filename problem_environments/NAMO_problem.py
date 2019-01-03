from manipulation.bodies.bodies import box_body, place_xyz_body
from manipulation.problems.problem import *
from manipulation.primitives.transforms import get_point, set_point, pose_from_quat_point, unit_quat
from manipulation.constants import PARALLEL_LEFT_ARM, REST_LEFT_ARM, HOLDING_LEFT_ARM, FOLDED_LEFT_ARM, \
    FAR_HOLDING_LEFT_ARM, LOWER_TOP_HOLDING_LEFT_ARM, REGION_Z_OFFSET
from manipulation.regions import create_region, AARegion
from manipulation.primitives.utils import mirror_arm_config
from manipulation.primitives.transforms import trans_from_base_values, set_pose, set_quat, \
    point_from_pose, axis_angle_from_rot, rot_from_quat, quat_from_pose, quat_from_z_rot, \
    get_pose, base_values_from_pose, pose_from_base_values, set_xy

import numpy as np
import copy
import sys
import os
import pickle

sys.path.append('../mover_library/')
from samplers import *
from utils import *
from operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp

# obj definitions
min_height = 0.4
max_height = 1

min_width = 0.2
max_width = 0.6

min_length = 0.2
max_length = 0.6

SLEEPTIME = 0.05


def create_objects(env, obj_region, table_region):
    #NUM_OBJECTS = 8
    NUM_OBJECTS = 3
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


def NAMO_problem(env):
    fdir=os.path.dirname(os.path.abspath(__file__))
    env.Load(fdir + '/namo_env.xml')
    robot = env.GetRobots()[0]
    set_point(env.GetKinBody('shelf1'), [1, -2.33205483, 0.010004])
    set_point(env.GetKinBody('shelf2'), [1, 2.33205483, 0.010004])

    set_config(robot, FOLDED_LEFT_ARM, robot.GetManipulator('leftarm').GetArmIndices())
    set_config(robot, mirror_arm_config(FOLDED_LEFT_ARM), robot.GetManipulator('rightarm').GetArmIndices())

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
    obj_region = AARegion('obj_region', ((-1.51 * 0.3, 2.51), (-2.51, 1.5)), z=0.0001, color=np.array((1, 1, 0, 0.25)))
    table_region = AARegion('table_region', ((-2.51 * 0.1, 2.51), (-2.51, -1)), z=0.0001,
                            color=np.array((1, 0, 1, 0.25)))
    entire_region = AARegion('entire_region', ((-2.51, 2.51), (-2.51, 2.51)), z=0.0001, color=np.array((1, 1, 0, 0.25)))
    loading_region = AARegion('loading_area',
                              ((-2.51, -0.81), (-2.51, 2.51)),
                              z=0.0001, color=np.array((1, 1, 0, 0.25)))
    OBJECTS, obj_poses, obj_shapes = create_objects(env, obj_region, table_region)

    # compute swept volume
    target_obj = env.GetKinBody('obj0')
    set_color(target_obj, (1, 0, 0))
    target_obj_shape = obj_shapes[target_obj.GetName()]

    initial_saver = DynamicEnvironmentStateSaver(env)

    problem = {'initial_saver': initial_saver,
               'robot_initial_config': robot_initial_config,
               'objects': OBJECTS,
               'obj_region': obj_region,
               'table_region': table_region,
               'loading_region': loading_region,
               'entire_region_xy': [0, 0],
               'entire_region_extents': [2.51, 2.51],
               'entire_region': entire_region,
               'env': env,
               'obj_shapes': obj_shapes,
               'obj_poses': obj_poses,
               'target_obj': target_obj}

    # order objects according to MCR?
    return problem
