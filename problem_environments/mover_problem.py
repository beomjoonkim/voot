from openravepy import *

import numpy as np
import sys


sys.path.append('../mover_library/')
from manipulation.constants import PARALLEL_LEFT_ARM, REST_LEFT_ARM, HOLDING_LEFT_ARM, FOLDED_LEFT_ARM, \
    FAR_HOLDING_LEFT_ARM, LOWER_TOP_HOLDING_LEFT_ARM, REGION_Z_OFFSET

from utils import *
from samplers import sample_pick
from manipulation.bodies.bodies import place_body, place_body_on_floor
#from utils import randomly_place_region
from manipulation.primitives.transforms import set_point
from manipulation.regions import create_region, AARegion
from motion_planner import collision_fn, base_extend_fn, base_sample_fn, base_distance_fn, smooth_path, rrt_connect

import os
import random

# obj definitions
min_height = 0.4
max_height = 1

min_width = 0.2
max_width = 0.6

min_length = 0.2
max_length = 0.6

OBST_COLOR = (1, 0, 0)
OBST_TRANSPARENCY = .25


def randomly_place_region(env, body, region):
    if env.GetKinBody(get_name(body)) is None:
        env.Add(body)
    while True:
        set_quat(body, quat_from_z_rot(uniform(0, 2 * PI)))
        aabb = aabb_from_body(body)
        cspace = region.cspace(aabb)
        if cspace is None: continue
        set_point(body, np.array([uniform(*range) for range in cspace] + [region.z + aabb.extents()[2] + BODY_PLACEMENT_Z_OFFSET]) - aabb.pos() + get_point(body))
        if not body_collision(env, body):
            return

def generate_rand(min, max):
    return np.random.rand() * (max - min) + min


def create_objects(env, obj_region, n_objects):
    OBJECTS = []
    obj_shapes = {}
    obj_poses = {}
    for i in range(n_objects):
        width = np.random.rand(1) * (max_width - min_width) + min_width
        length = np.random.rand(1) * (max_width - min_length) + min_length
        height = np.random.rand(1) * (max_height - min_height) + min_height
        new_body = box_body(env, width, length, height,
                            name='obj%s' % i,
                            color=(0, (i + .5) / n_objects, 0))
        trans = np.eye(4)
        trans[2, -1] = 0.075
        env.Add(new_body)
        new_body.SetTransform(trans)
        obj_pose = randomly_place_region(env, new_body, obj_region)  # TODO fix this to return xytheta
        OBJECTS.append(new_body)

        obj_shapes['obj%s' % i] = [width[0], length[0], height[0]]
        obj_poses['obj%s' % i] = obj_pose
    return OBJECTS, obj_poses, obj_shapes


def load_objects(env, obj_shapes, obj_poses, color):
    # sets up the object at their locations in the original env
    OBJECTS = []
    i = 0
    nobj = len(obj_shapes.keys())
    for obj_name in obj_shapes.keys():
        obj_xyz = obj_poses[obj_name]['obj_xyz']
        obj_rot = obj_poses[obj_name]['obj_rot']
        width, length, height = obj_shapes[obj_name]

        new_body = box_body(env, width, length, height,
                            name=obj_name,
                            color=np.array(color) / float(nobj - i))
        i += 1
        env.Add(new_body)
        set_point(new_body, obj_xyz)
        set_quat(new_body, obj_rot)
        OBJECTS.append(new_body)
    return OBJECTS


def create_bottom_walls(x_lim, y_lim, env):
    bottom_wall = box_body(env, x_lim * 2, y_lim * 2, 0.135, name='bottom_wall', color=(.82, .70, .55))
    bottom_wall_x = x_lim / 2.0
    set_xy(bottom_wall, bottom_wall_x, 0)
    env.Add(bottom_wall)

    side_wall = box_body(env, y_lim * 2, 0.01 * 2, 0.2 * 2,
                         name='side_wall_1',
                         color=(.82, .70, .55))
    place_body(env, side_wall, (-x_lim + bottom_wall_x, 0, np.pi / 2), base_name='bottom_wall')

    side_wall = box_body(env, y_lim * 2, 0.01 * 2, 0.2 * 2,
                         name='side_wall_2',
                         color=(.82, .70, .55))
    place_body(env, side_wall, (x_lim + bottom_wall_x, 0, np.pi / 2), base_name='bottom_wall')

    side_wall = box_body(env, x_lim * 2, 0.01 * 2, 0.2 * 2,
                         name='side_wall_3',
                         color=(.82, .70, .55))
    place_body(env, side_wall, (bottom_wall_x, y_lim, 0), base_name='bottom_wall')

    entrance_width = 1.2
    left_door_length = (x_lim - entrance_width / 2.0) - 1
    right_door_length = (x_lim - entrance_width / 2.0) + 1
    entrance_left = box_body(env, left_door_length, 0.01 * 2, 0.2 * 2,
                             name='entrance_left',
                             color=(.82, .70, .55))
    entrance_right = box_body(env, right_door_length, 0.01 * 2, 0.2 * 2,
                              name='entrance_right',
                              color=(.82, .70, .55))

    place_body(env, entrance_left, (bottom_wall_x + (-x_lim - (entrance_width / 2.0 + 1)) / 2.0, -y_lim, 0),
               base_name='bottom_wall')
    place_body(env, entrance_right, (bottom_wall_x + (x_lim + (entrance_width / 2.0 - 1)) / 2.0, -y_lim, 0),
               base_name='bottom_wall')

    # place_body(env, env.GetKinBody('Cabinet'), (x_lim + x_lim / 3.0 - 0.5, y_lim - 3, np.pi), base_name='bottom_wall')


def create_doors(x_lim, y_lim, door_x, door_y, door_width, th, env):
    if th == 0:
        right_wall_size = (door_y - door_width / 2.0 - (-y_lim)) / 2.0
        left_wall_size = (y_lim - door_width / 2.0 - door_y) / 2.0
    elif th == np.pi / 2.:
        right_wall_size = (door_x - door_width / 2.0 - (-x_lim)) / 2.0
        left_wall_size = (x_lim - door_width / 2.0 - door_x) / 2.0

    left_wall = box_body(env,
                         0.04 * 2, left_wall_size * 2, 1 * 2,
                         name='left_wall',
                         color=(0, 0, 0))
    right_wall = box_body(env,
                          0.04 * 2, right_wall_size * 2, 1 * 2,
                          name='right_wall',
                          color=(0, 0, 1))
    if th == 0:
        place_body(env, left_wall, (door_x, door_y + left_wall_size + (door_width / 2.), th),
                   base_name='bottom_wall')
        place_body(env, right_wall, (door_x, door_y - right_wall_size - (door_width / 2.), th),
                   base_name='bottom_wall')
    else:
        place_body(env, left_wall, (door_x + left_wall_size + (door_width / 2.), door_y, th),
                   base_name='bottom_wall')
        place_body(env, right_wall, (door_x - right_wall_size - (door_width / 2.), door_y, th),
                   base_name='bottom_wall')


def create_box_bodies(body_shape, color, name, n_objs, env):
    if color == 'green':
        box_bodies = [box_body(env, body_shape[i][0], body_shape[i][1], body_shape[i][2], name=name + '%s' % i,
                               color=(1, (i + .5) / 5, 0)) for i in range(n_objs)]
    elif color == 'red':
        box_bodies = [box_body(env, body_shape[i][0], body_shape[i][1], body_shape[i][2], name=name + '%s' % i,
                               color=((i + .5) / 5, 0, 1)) for i in range(n_objs)]
    elif color == 'blue':
        box_bodies = [box_body(env, body_shape[i][0], body_shape[i][1], body_shape[i][2], name=name + '%s' % i,
                               color=(0, 1, (i + .5) / 5)) for i in range(n_objs)]

    return box_bodies


def generate_shelf_shapes():
    max_shelf_width = 0.7
    min_shelf_width = 0.4

    center_shelf_width = generate_rand(min_shelf_width, max_shelf_width)  # np.random.rand(0.5,0.8)
    right_shelf_width = generate_rand(min_shelf_width, max_shelf_width)
    left_shelf_width = generate_rand(min_shelf_width, max_shelf_width)

    center_shelf_height = generate_rand(0.3, 0.5)
    center_shelf_top_height = generate_rand(0.3, 0.5)
    left_shelf_height = generate_rand(0.3, 0.5)
    left_shelf_top_height = generate_rand(0.3, 0.5)
    right_shelf_height = generate_rand(0.3, 0.5)
    right_shelf_top_height = generate_rand(0.3, 0.5)

    shelf_shapes = {'center_shelf_top_height': center_shelf_top_height,
                    'center_shelf_height': center_shelf_height,
                    'left_shelf_top_height': left_shelf_top_height,
                    'left_shelf_height': left_shelf_height,
                    'right_shelf_top_height': right_shelf_top_height,
                    'right_shelf_height': right_shelf_height,
                    'center_shelf_width': center_shelf_width,
                    'left_shelf_width': left_shelf_width,
                    'right_shelf_width': right_shelf_width}

    left_x = center_shelf_width / 2.0 + left_shelf_width / 2.0
    right_x = -center_shelf_width / 2.0 - right_shelf_width / 2.0  # center of left shelf

    shelf_xs = {
        'left_x': left_x,
        'right_x': right_x
    }
    return shelf_shapes, shelf_xs


def create_shelf(env, obst_x, obst_width, obst_height, name_idx, stacked_obj_name, table_name):
    width = 0.4
    length = 0.01
    height = obst_height
    top_wall_width = 0.001
    bottom_wall_width = 0.0001

    table_pos = aabb_from_body(env.GetKinBody(table_name)).pos()
    table_x = table_pos[0]
    table_y = table_pos[1]
    place_body(env,
               box_body(env,
                        width, length, height,
                        name='right_wall_' + str(name_idx),
                        color=OBST_COLOR,
                        transparency=OBST_TRANSPARENCY),
               (table_x + .0, table_y + obst_x - (obst_width - .05) / 2, 0),
               stacked_obj_name)
    place_body(env,
               box_body(env,
                        width, length, height,
                        name='left_wall_' + str(name_idx),
                        color=OBST_COLOR,
                        transparency=OBST_TRANSPARENCY),
               (table_x + .0, table_y + obst_x + (obst_width - .05) / 2, 0),
               stacked_obj_name)
    place_body(env,
               box_body(env,
                        length, obst_width - 0.05, height,
                        name='back_wall_' + str(name_idx),
                        color=OBST_COLOR,
                        transparency=OBST_TRANSPARENCY),
               (table_x + .225, table_y + obst_x, 0),
               stacked_obj_name)
    place_body(env,
               box_body(env,
                        width, obst_width - 0.05, top_wall_width,
                        name='top_wall_' + str(name_idx),
                        color=OBST_COLOR,
                        transparency=OBST_TRANSPARENCY),
               (table_x + 0, table_y + obst_x, 0),
               'back_wall_' + str(name_idx))
    place_body(env,
               box_body(env,
                        width, obst_width - 0.05, bottom_wall_width,
                        name='bottom_wall_' + str(name_idx),
                        color=OBST_COLOR,
                        transparency=0.5),
               (table_x + 0, table_y + obst_x, 0),
               stacked_obj_name)

    region = create_region(env, 'place_region_' + str(name_idx),
                           ((-1.0, 1.0), (-0.85, 0.85)),
                           'bottom_wall_' + str(name_idx), color=np.array((0, 0, 0, .5)))
    region.draw(env)
    return region


def set_fixed_object_poses(env, x_lim, y_lim):
    objects = [env.GetKinBody('shelf1'), env.GetKinBody('shelf2'), env.GetKinBody('computer_table'),
               env.GetKinBody('table2')]
    place_body(env, env.GetKinBody('shelf1'), (x_lim + x_lim / 2.0 - 0.5, y_lim - 0.2, np.pi * 3 / 2),
               base_name='bottom_wall')
    place_body(env, env.GetKinBody('shelf2'), (x_lim + x_lim / 2.0 - 1.5, y_lim - 0.2, np.pi * 3 / 2),
               base_name='bottom_wall')
    place_body(env, env.GetKinBody('table2'), (x_lim + x_lim / 2.0 - 0.5, y_lim - 3, np.pi * 3 / 2),
               base_name='bottom_wall')
    place_body(env, env.GetKinBody('computer_chair'), (4.2, -1.5, 0), base_name='bottom_wall')
    obj_poses = {obj.GetName(): get_pose(obj) for obj in objects}
    return obj_poses


def create_shelves(env, shelf_shapes, shelf_xs, table_name):
    center_shelf_width = shelf_shapes['center_shelf_width']
    center_shelf_height = shelf_shapes['center_shelf_height']
    #left_shelf_width = shelf_shapes['left_shelf_width']
    #left_shelf_height = shelf_shapes['left_shelf_height']
    right_shelf_width = shelf_shapes['right_shelf_width']
    right_shelf_height = shelf_shapes['right_shelf_height']
    center_shelf_top_height = shelf_shapes['center_shelf_top_height']
    left_shelf_top_height = shelf_shapes['left_shelf_top_height']
    right_shelf_top_height = shelf_shapes['right_shelf_top_height']

    left_x = shelf_xs['left_x']
    right_x = shelf_xs['right_x']

    center_region = create_shelf(env, obst_x=0, obst_width=center_shelf_width,
                                 obst_height=center_shelf_height, name_idx=1, stacked_obj_name=table_name,
                                 table_name=table_name)
    center_top_region = create_shelf(env, obst_x=0, obst_width=center_shelf_width,
                                     obst_height=center_shelf_top_height, name_idx=2,
                                     stacked_obj_name='back_wall_1', table_name=table_name)

    """
    left_region = create_shelf(env, obst_x=left_x, obst_width=left_shelf_width,
                               obst_height=left_shelf_height, name_idx=3, stacked_obj_name=table_name,
                               table_name=table_name)
    left_top_region = create_shelf(env, obst_x=left_x, obst_width=left_shelf_width,
                                   obst_height=left_shelf_top_height, name_idx=4,
                                   stacked_obj_name='back_wall_3', table_name=table_name)
    """
    right_region = create_shelf(env, obst_x=right_x, obst_width=right_shelf_width,
                                obst_height=right_shelf_height, name_idx=5, stacked_obj_name=table_name,
                                table_name=table_name)
    right_top_region = create_shelf(env, obst_x=right_x, obst_width=right_shelf_width,
                                    obst_height=right_shelf_top_height, name_idx=6,
                                    stacked_obj_name='back_wall_5', table_name=table_name)
    regions = {'center': center_region, 'center_top': center_top_region,
    #           'left': left_region, 'left_top': left_top_region}
               'right': right_region, 'right_top': right_top_region}
    return regions


def generate_shelf_obj_shapes():
    max_obj_height = 0.28
    min_obj_height = 0.15
    n_objs = 3
    l_obj_shapes = [[0.05, 0.05, generate_rand(min_obj_height, max_obj_height)] for _ in range(n_objs)]
    ltop_obj_shapes = [[0.05, 0.05, generate_rand(min_obj_height, max_obj_height)] for _ in range(n_objs)]
    c_obj_shapes = [[0.05, 0.05, generate_rand(min_obj_height, max_obj_height)] for _ in range(n_objs)]
    ctop_obj_shapes = [[0.05, 0.05, generate_rand(min_obj_height, max_obj_height)] for _ in range(n_objs)]
    r_obj_shapes = [[0.05, 0.05, generate_rand(min_obj_height, max_obj_height)] for _ in range(n_objs)]
    rtop_obj_shapes = [[0.05, 0.05, generate_rand(min_obj_height, max_obj_height)] for _ in range(n_objs)]

    obj_shapes = {'l_obj_shapes': l_obj_shapes, 'ltop_obj_shapes': ltop_obj_shapes,
                  'c_obj_shapes': c_obj_shapes, 'ctop_obj_shapes': ctop_obj_shapes,
                  'r_obj_shapes': r_obj_shapes, 'rtop_obj_shapes': rtop_obj_shapes}

    return obj_shapes


def create_shelf_objs(env, obj_shapes):
    n_objs = 3
    #left_objs = create_box_bodies(obj_shapes['l_obj_shapes'], color='green', name='l_obst', n_objs=n_objs,
    #                              env=env)
    #left_top_objs = create_box_bodies(obj_shapes['ltop_obj_shapes'], color='green', name='ltop_obst',
    #                                  n_objs=n_objs, env=env)
    center_objs = create_box_bodies(obj_shapes['c_obj_shapes'], color='blue', name='c_obst', n_objs=n_objs,
                                    env=env)
    center_top_objs = create_box_bodies(obj_shapes['ctop_obj_shapes'], color='blue', name='ctop_obst',
                                        n_objs=n_objs, env=env)
    right_objs = create_box_bodies(obj_shapes['r_obj_shapes'], color='red', name='r_obst',
                                   n_objs=n_objs, env=env)
    right_top_objs = create_box_bodies(obj_shapes['rtop_obj_shapes'], color='red', name='rtop_obst',
                                       n_objs=n_objs, env=env)
    objects = {#'left': left_objs, 'left_top': left_top_objs,
               'center': center_objs, 'center_top': center_top_objs,
               'right': right_objs, 'right_top': right_top_objs}
    return objects


def place_objs_in_region(objs, region, env):
    for obj in objs:
        randomly_place_region(env, obj, region)


def generate_poses_and_place_shelf_objs(objects, regions, env):
    """
    left_objs = objects['left']
    left_region = regions['left']
    left_top_objs = objects['left_top']
    left_top_region = regions['left_top']
    """
    right_objs = objects['right']
    right_region = regions['right']
    right_top_objs = objects['right_top']
    right_top_region = regions['right_top']
    center_objs = objects['center']
    center_region = regions['center']
    center_top_objs = objects['center_top']
    center_top_region = regions['center_top']

    #place_objs_in_region(left_objs, left_region, env)
    #place_objs_in_region(left_top_objs, left_top_region, env)
    place_objs_in_region(center_objs, center_region, env)
    place_objs_in_region(center_top_objs, center_top_region, env)
    place_objs_in_region(right_objs, right_region, env)
    place_objs_in_region(right_top_objs, right_top_region, env)

    obj_poses = {obj.GetName(): get_pose(obj) for obj_list in objects.values() for obj in obj_list}
    return obj_poses


def set_fixed_obj_poses(env):
    table1 = env.GetKinBody('table1')
    shelf = env.GetKinBody('shelf1')


def create_environment_region(name, xy, extents, z=None):
    if z is None:
        z = 0.138

    region = AARegion(name, ((xy[0] - extents[0],
                                 xy[0] + extents[0]),
                                (xy[1] - extents[1],
                                 xy[1] + extents[1])),
             z, color=np.array((1, 1, 0, 0.25)))
    return region


def mover_problem(env):
    x_extents = 3.5
    y_extents = 3.16
    door_width = 1.5  # generate_rand(1, 1.5)
    door_x = (-x_extents + 1.5 + x_extents - 1.5) / 2.0 - x_extents * 0.3 + 4
    door_y = (-y_extents + 1.5 + y_extents - 1.5) / 2.0
    door_th = 0

    # todo move all the kitchen objects by 0.5

    fdir = os.path.dirname(os.path.abspath(__file__))
    env.Load(fdir + '/resources/mover_env.xml')
    # set_xy(env.GetKinBody('kitchen'), 0, 0.5)

    robot = env.GetRobots()[0]
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
                                                                  iktype=IkParameterization.Type.Transform6D)
                                                                  #forceikfast=True, freeindices=None,
                                                                  #freejoints=None, manip=None)
    if not ikmodel2.load():
        ikmodel2.autogenerate()


    create_bottom_walls(x_extents, y_extents, env)
    create_doors(x_extents, y_extents, door_x, door_y, door_width, door_th, env)
    set_config(robot, FOLDED_LEFT_ARM, robot.GetManipulator('leftarm').GetArmIndices())
    set_config(robot, mirror_arm_config(FOLDED_LEFT_ARM), robot.GetManipulator('rightarm').GetArmIndices())
    fixed_obj_poses = set_fixed_object_poses(env, x_extents, y_extents)
    shelf_shapes, shelf_xs = generate_shelf_shapes()
    shelf_regions = create_shelves(env, shelf_shapes, shelf_xs, 'table2')
    env.SetViewer('qtcoin')

    obj_shapes = generate_shelf_obj_shapes()
    shelf_objects = create_shelf_objs(env, obj_shapes)
    shelf_obj_poses = generate_poses_and_place_shelf_objs(shelf_objects, shelf_regions, env)
    for region_name, region in zip(shelf_regions.keys(), shelf_regions.values()):
        region.name = region_name+'_shelf_region'


    home_region_xy = [x_extents/2.0, 0]
    home_region_xy_extents = [x_extents, y_extents]
    home_region = AARegion('home_region',((-x_extents + x_extents / 2.0, x_extents + x_extents / 2.0), (-y_extents, y_extents)), z=0.135, color=np.array((1, 1, 0, 0.25)))

    loading_region_xy = [1.8, -6.7]
    loading_region_xy_extents = [2.5, 1.85]
    loading_region = AARegion('loading_region', ((loading_region_xy[0] - loading_region_xy_extents[0],
                                                  loading_region_xy[0] + loading_region_xy_extents[0]),
                                                 (loading_region_xy[1] - loading_region_xy_extents[1],
                                                  loading_region_xy[1] + loading_region_xy_extents[1])),
                              z=0.138, color=np.array((1, 1, 0, 0.25)))
    bridge_region_name = 'bridge_region'
    bridge_region_xy = [0.7, -4.1]
    bridge_region_extents = [1, 1.0]
    bridge_region = create_environment_region(bridge_region_name, bridge_region_xy, bridge_region_extents)

    entire_region_xy = [x_extents/2.0, -3]
    entire_region_xy_extents = [x_extents, y_extents+2.5]
    entire_region = AARegion('entire_region',((-entire_region_xy_extents[0] + entire_region_xy[0], entire_region_xy_extents[0] + entire_region_xy[0]), (-entire_region_xy_extents[1]+entire_region_xy[1], entire_region_xy_extents[1]+entire_region_xy[1])), z=0.135, color=np.array((1, 1, 0, 0.25)))

    packing_boxes = [b for b in env.GetBodies() if b.GetName().find('packing_box') != -1]

    #place_objs_in_region(packing_boxes, loading_region, env)
    #place_objs_in_region([robot], loading_region, env)

    place_objs_in_region(packing_boxes[0:5], home_region, env)
    place_objs_in_region(packing_boxes[5:], home_region, env)
    place_objs_in_region([robot], loading_region, env)
    #xy = get_body_xytheta(env.GetKinBody('rectangular_packing_box1'))[0,0:2]
    #set_obj_xytheta(np.array([2.46, 1.3244, 0]), env.GetKinBody('rectangular_packing_box1'))
    set_obj_xytheta(np.array([-0.46, -1.3244, 0]), robot)

    box_regions = {}
    for box in packing_boxes:
        box_region = AARegion.create_on_body(box)
        box_region.color = (1., 1., 0., 0.25)
        box_regions[box.GetName()] = box_region
        if box == packing_boxes[0]:
            xytheta = get_body_xytheta(box)
            set_obj_xytheta([xytheta[0,0],xytheta[0,1],0],box)
            box_region.draw(env)
    open_gripper(robot)
    temp_objects_to_pack = [body for body in env.GetBodies() if
                       body.GetName().find('box') == -1 and body.GetName().find('wall') == -1 and
                       body.GetName().find('sink') == -1 and body.GetName().find('kitchen') == -1 and
                       body.GetName().find('entrance') == -1 and body.GetName().find('pr2') == -1 and
                       body.GetName().find('floorwalls') == -1 and body.GetName().find('table') == -1 and
                       body.GetName().find('obst') == -1]


    # packing boxes are packed in the order given in packing_boxes
    # 1. packing boxes in the home
    # 2. big objects in the truck
    # 3. small objects in the box
    # 4. shelf objects in the box
    # 5. boxes in the truck

    big_objects_to_pack = [body for body in env.GetBodies()
                           if body.GetName().find('chair') != -1 or body.GetName().find('shelf') !=-1]

    objects_to_pack = [obj for obj in temp_objects_to_pack if obj not in big_objects_to_pack]
    problem_config = {'shelf_objects': shelf_objects,
                      'packing_boxes': packing_boxes,
                      'objects_to_pack': objects_to_pack,
                      'big_objects_to_pack': big_objects_to_pack,
                      'home_region': home_region,
                      'loading_region': loading_region,
                      'entire_region': entire_region,
                      'entire_region_xy': entire_region_xy,
                      'entire_region_extents': entire_region_xy_extents,
                      'bridge_region': bridge_region,
                      'bridge_region_xy': bridge_region_xy,
                      'bridge_region_extents': bridge_region_extents,
                      'env': env,
                      'loading_region_xy': loading_region_xy,
                      'loading_region_extents': loading_region_xy_extents,
                      'home_region_xy': home_region_xy,
                      'home_region_extents': home_region_xy_extents,
                      'shelf_regions': shelf_regions,
                      'box_regions': box_regions}

    return problem_config
