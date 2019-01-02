from manipulation.primitives.transforms import get_point, set_point
from manipulation.bodies.bodies import set_config
from manipulation.bodies.bodies import box_body, randomly_place_body, place_xyz_body
from manipulation.motion.primitives import extend_fn, distance_fn, sample_fn, collision_fn

from openravepy import *
import sys
import os
import pickle
import numpy as np
from sklearn import preprocessing

from motion_planner import get_number_of_confs_in_between

def base_config_outside_threshold(c, configs, body):
    c = np.array(c)
    for cprime in configs:
        n_between = get_number_of_confs_in_between(c, cprime, body)
        if n_between <= 1:
            return False
    return True

"""
def base_config_outside_threshold(c, configs, xy_threshold, th_threshold):
    min_dist = np.inf
    c = np.array(c)
    for cprime in configs:
        cprime = np.array(cprime)
        xy_dist = np.linalg.norm(c[0:2] - cprime[0:2])
        # th_dist = abs(c[2]-cprime[2])
        th_dist = abs(c[2] - cprime[2]) if abs(c[2] - cprime[2]) < np.pi else 2 * np.pi - abs(c[2] - cprime[2])
        assert (th_dist < np.pi)

        if xy_dist < xy_threshold and th_dist < th_threshold:
            return False
    return True
"""


def leftarm_torso_config_outside_threshold(c, configs, threshold):
    """
    Original joint limits for leftarm_torso
    12 < joint:torso_lift_joint(12), dof = 12, parent = pr2 > [0.0115 0.305]
    15 < joint:l_shoulder_pan_joint(15), dof = 15, parent = pr2 > [-0.5646018   2.13539289]
    16 < joint:l_shoulder_lift_joint(16), dof = 16, parent = pr2 > [-0.35360022  1.29629967]
    17 < joint:l_upper_arm_roll_joint(17), dof = 17, parent = pr2 > [-0.65000076  3.74999698]
    18 < joint:l_elbow_flex_joint(18), dof = 18, parent = pr2 > [-2.12130808 - 0.15000005]
    19 < joint:l_forearm_roll_joint(19), dof = 19, parent = pr2 > [-10000.  10000.]
    20 < joint:l_wrist_flex_joint(20), dof = 20, parent = pr2 > [-2.0000077 - 0.10000004]
    21 < joint:l_wrist_roll_joint(21), dof = 21, parent = pr2 > [-10000.  10000.]
    """

    config_lower_limit = np.array([0.0115, -0.5646018, -0.35360022, -0.65000076, -2.12130808, -3.14159265, -2.0000077,
                                   -3.14159265]),
    config_upper_limit = np.array([0.305, 2.13539289, 1.29629967, 3.74999698, -0.15000005, 3.14159265,
                                   -0.10000004,  3.14159265])
    #weights = 1.0/(config_upper_limit - config_lower_limit)

    each_config_threshold = (config_upper_limit - config_lower_limit)*threshold
    for cprime in configs:
        diff = abs(cprime-c)
        if np.all(diff < each_config_threshold):
            return False
    return True


def get_configs_from_paths(paths, normalized_paths, configs, normalized_configs, body):
    for path, normalized_path in zip(paths, normalized_paths):
        for c, normalized_c in zip(path, normalized_path):
            if abs(c[-1]) > np.pi:
                import pdb;pdb.set_trace()
            if base_config_outside_threshold(c, configs, body):
                configs.append(c)
                normalized_configs.append(normalized_c)


"""
def get_configs_from_paths(paths, configs, xy_threshold=0.1, th_threshold=20*np.pi/180):
    for path in paths:
        for c in path:
            if c[-1] < 0:
                c[-1] += 2 * np.pi
            if c[-1] > 2 * np.pi:
                c[-1] -= 2 * np.pi
            if base_config_outside_threshold(c, configs, xy_threshold, th_threshold):
                configs.append(c)
"""


def get_leftarm_torso_configs_from_paths(paths, configs, threshold):
    for path in paths:
        for c in path:
            if leftarm_torso_config_outside_threshold(c, configs, threshold):
                configs.append(c)


def get_paths(nodes):
    paths = [n.sample['path'] for n in nodes if (n.sample != None) \
             and ('path' in n.sample.keys())]
    paths = []
    for n in nodes:
        if n.sample == None: continue
        if not 'path' in n.sample.keys(): continue
        raw_path = n.sample['path']
        stepsize = int(round(0.1 * len(raw_path), 0))
        idxs = range(1, len(raw_path), stepsize)
        path = [raw_path[idx] for idx in idxs]
        if not np.all(path[-1] == raw_path[-1]):
            path.append(raw_path[-1])
        paths.append(path)
    return paths
