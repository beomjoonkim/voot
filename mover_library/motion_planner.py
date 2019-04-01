from motion_planners.rrt import TreeNode, configs
from motion_planners.utils import argmin
import numpy as np
from random import randint


def get_number_of_base_confs_in_between(q1, q2, body):
    resolution = np.array([0.2, 0.2, 10 * np.pi / 180.0])
    n = int(
        np.max(np.abs(np.divide(body.SubtractActiveDOFValues(q1, q2), resolution)))) + 1
    return n


def leftarm_torso_linear_interpolation(body, q1, q2, resolution):  # Sequence doesn't include q1
    """
    config_lower_limit = np.array([0.0115, -0.5646018, -0.35360022, -0.65000076, -2.12130808, -3.14159265, -2.0000077,
                                   -3.14159265]),
    config_upper_limit = np.array([0.305, 2.13539289, 1.29629967, 3.74999698, -0.15000005, 3.14159265,
                                   -0.10000004,  3.14159265])
    resolution = (config_upper_limit - config_lower_limit) * resolution # a portion of the configuration limits
    n = int(np.max(np.abs(np.divide(body.SubtractActiveDOFValues(q2, q1), resolution)))) + 1
    """
    n = int(np.max(np.abs(np.divide(body.SubtractActiveDOFValues(q2, q1), body.GetActiveDOFResolutions())))) + 1
    # If the resolution for a particular joint angle is say, 0.02, then we are assuming that within the 0.02 of the
    # angle value, there would not be a collision, or even if there is, we are going to ignore it.
    q = q1
    for i in range(n):
        q = (1. / (n - i)) * body.SubtractActiveDOFValues(q2, q) + q  # NOTE - do I need to repeatedly do the subtract?
        yield q


def leftarm_torso_extend_fn(body, resolution=0.05):
    return lambda q1, q2: leftarm_torso_linear_interpolation(body, q1, q2, resolution)


def collision_fn(env, body, check_self=False):
    def fn(q):
        with body:
            body.SetActiveDOFValues(q)
            return env.CheckCollision(body) or (check_self and body.CheckSelfCollision())
    return fn


def extend_fn(body):
    return lambda q1, q2: linear_interpolation(body, q1, q2)


def base_extend_fn(body):
    return lambda q1, q2: base_linear_interpolation(body, q1, q2)


def arm_base_extend_fn(body):
    return lambda q1, q2: arm_base_linear_interpolation(body, q1, q2)


def sample_fn(body, collisions=False):
    return lambda: cspace_sample(body) if not collisions else cspace_sample_collisions(body)


def arm_base_sample_fn(body, x_extents, y_extents, x=0, y=0):
    return lambda: base_arm_cspace_sample(body, x_extents, y_extents, x, y)


def base_sample_fn(body, x_extents, y_extents, x=0, y=0):  # body is passed in for consistency
    return lambda: base_cspace_sample(x_extents, y_extents, x, y)


def distance_fn(body):
    return lambda q1, q2: cspace_distance_2(body, q1, q2)


def base_distance_fn(body, x_extents, y_extents):
    return lambda q1, q2: base_distance(q1, q2, x_extents, y_extents)


def arm_base_distance_fn(body, x_extents, y_extents):
    return lambda q1, q2: arm_base_cspace_distance_2(body, q1, q2, x_extents, y_extents)


def base_distance(q1, q2, x_extents, y_extents):
    distance = abs(q1-q2)
    if distance[-1] > np.pi:
        distance[-1] = 2*np.pi - distance[-1]
    return np.dot(distance, 1/np.array([2 * x_extents, 2 * y_extents, np.pi]))  # normalize them by their max values


def cspace_sample_collisions(body):
    while True:
        config = cspace_sample(body)
        body.SetActiveDOFValues(config)
        if not body.env().CheckCollision(body):  # NOTE - not thread-safe get rid of
            return config

def arm_base_cspace_distance_2(body, q1, q2, x_extents, y_extents):
    arm_diff = body.SubtractActiveDOFValues(q2, q1)[:-3]
    dim_arm = len(q1) - 3
    arm_weights = np.ones((dim_arm,)) * 2*np.pi
    arm_dist = np.dot(1./arm_weights, arm_diff*arm_diff)
    base_dist = base_distance(q1[-3:], q2[-3:], x_extents, y_extents)
    return base_dist+arm_dist


def cspace_distance_2(body, q1, q2):
    diff = body.SubtractActiveDOFValues(q2, q1)
    return np.dot(body.GetActiveDOFWeights(), diff * diff)


def base_cspace_sample(x_extents, y_extents, x, y):
    lower_lim = np.array([x-x_extents, y-y_extents, -np.pi])
    upper_lim = np.array([x+x_extents, y+y_extents, np.pi])
    return np.random.uniform(lower_lim, upper_lim)


def base_arm_cspace_sample(body, x_extents, y_extents, x, y):
    lower_lim = np.array([x-x_extents, y-y_extents, -np.pi])
    upper_lim = np.array([x+x_extents, y+y_extents, np.pi])
    base_config = np.random.uniform(lower_lim, upper_lim)
    arm_config = cspace_sample(body)[:-3]
    return np.hstack([arm_config, base_config])


def cspace_sample(body):
    return np.random.uniform(*body.GetActiveDOFLimits())  # TODO - adjust theta limits to be between [-PI, PI)


def arm_base_linear_interpolation(body, q1, q2):
    diff = body.SubtractActiveDOFValues(q2, q1)

    arm_resolution = body.GetActiveDOFResolutions()[:-3]*5
    base_resolution = np.array([0.2, 0.2, 20 * np.pi / 180.0])
    full_config_resolution = np.hstack([arm_resolution, base_resolution])
    n = int(np.max(np.abs(np.divide(diff, full_config_resolution )))) + 1
    q = q1
    for i in range(n):
        q = (1. / (n - i)) * body.SubtractActiveDOFValues(q2, q) + q  # NOTE - do I need to repeatedly do the subtract?
        yield q


def linear_interpolation(body, q1, q2):  # Sequence doesn't include q1
    n = int(np.max(np.abs(np.divide(body.SubtractActiveDOFValues(q2, q1), body.GetActiveDOFResolutions()*10 )))) + 1
    q = q1
    for i in range(n):
        q = (1. / (n - i)) * body.SubtractActiveDOFValues(q2, q) + q  # NOTE - do I need to repeatedly do the subtract?
        yield q

"""
def base_linear_interpolation(body, q1, q2):
    n = int(
        np.max(np.abs(np.divide(body.SubtractActiveDOFValues(q2, q1), np.array([0.2, 0.2, 20 * np.pi / 180.0]))))) + 1
    q = q1
    interpolated_qs = []
    for i in range(n):
        q = (1. / (n - i)) * body.SubtractActiveDOFValues(q2, q) + q  # NOTE - do I need to repeatedly do the subtract?
        interpolated_qs.append(q)
    return interpolated_qs
"""

def base_linear_interpolation(body, q1, q2):
    n = get_number_of_base_confs_in_between(q1, q2, body)
    q = q1
    interpolated_qs = []
    for i in range(n):
        curr_q = q
        q = (1. / (n - i)) * body.SubtractActiveDOFValues(q2, curr_q) + curr_q  # NOTE - do I need to repeatedly do the subtract?
        if q[-1] > np.pi:
            q[-1] = q[-1] - 2*np.pi
        if q[-1] < -np.pi:
            q[-1] = q[-1] + 2*np.pi
        interpolated_qs.append(q)
    return interpolated_qs


def rrt_connect(q1, q2, distance, sample, extend, collision, iterations):
    # check if q1 or q2 is in collision
    if collision(q1) or collision(q2):
        print 'collision in either initial or goal'
        return None

    # define two roots of the tree
    root1, root2 = TreeNode(q1), TreeNode(q2)

    # nodes1 grows from q1, nodes2 grows from q2
    nodes1, nodes2 = [root1], [root2]

    # sample and extend iterations number of times
    for ntry in range(iterations):
        if len(nodes1) > len(nodes2):  # ????
            nodes1, nodes2 = nodes2, nodes1

        # sample a configuration
        s = sample()

        # returns the node with the closest distance to s from a set of nodes nodes1
        last1 = argmin(lambda n: distance(n.config, s), nodes1)

        # extend from the closest config to s
        for q in extend(last1.config, s):  # I think this is what is taking up all the time
            # if there is a collision, extend upto that collision
            if collision(q):
                break
            last1 = TreeNode(q, parent=last1)
            nodes1.append(last1)

        # try to extend to the tree grown from the other side
        last2 = argmin(lambda n: distance(n.config, last1.config), nodes2)

        for q in extend(last2.config, last1.config):
            if collision(q):
                break
            last2 = TreeNode(q, parent=last2)
            nodes2.append(last2)
        else:  # where is the if for this else?
            # apparently, if none of q gets into
            # if  collision(q) stmt, then it will enter here
            # two trees meet at the new configuration s
            path1, path2 = last1.retrace(), last2.retrace()
            if path1[0] != root1:
                path1, path2 = path2, path1
            return configs(path1[:-1] + path2[::-1])
    return None


def direct_path(q1, q2, extend, collision):
    path = [q1]
    for q in extend(q1, q2):
        if collision(q):
            return None
        path.append(q)
    return path


def smooth_path(path, extend, collision, iterations=50):
    smoothed_path = path
    for _ in range(iterations):
        if len(smoothed_path) <= 2:
            return smoothed_path
        i, j = randint(0, len(smoothed_path) -
                       1), randint(0, len(smoothed_path) - 1)
        if abs(i - j) <= 1:
            continue
        if j < i:
            i, j = j, i
        shortcut = list(extend(smoothed_path[i], smoothed_path[j]))
        if len(shortcut) < j - i and all(not collision(q) for q in shortcut):
            smoothed_path = smoothed_path[
                            :i + 1] + shortcut + smoothed_path[j + 1:]
    return smoothed_path

