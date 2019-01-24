from manipulation.problems.problem import *
from manipulation.bodies.bodies import get_name

from manipulation.primitives.transforms import get_point, set_point, pose_from_quat_point, unit_quat
from manipulation.constants import BODY_PLACEMENT_Z_OFFSET
from manipulation.constants import *

##TODO: Clean this
from manipulation.constants import PARALLEL_LEFT_ARM, REST_LEFT_ARM, HOLDING_LEFT_ARM, FOLDED_LEFT_ARM, \
    FAR_HOLDING_LEFT_ARM, LOWER_TOP_HOLDING_LEFT_ARM, REGION_Z_OFFSET
from manipulation.regions import create_region, AARegion

from manipulation.primitives.transforms import trans_from_base_values, set_pose, set_quat, \
    point_from_pose, axis_angle_from_rot, rot_from_quat, quat_from_pose, quat_from_z_rot, \
    get_pose, base_values_from_pose, pose_from_base_values, set_xy, quat_from_angle_vector, \
    quat_from_trans

from utils import *
import numpy as np

from manipulation.bodies.bounding_volumes import aabb_from_body

# search episode

from manipulation.primitives.inverse_kinematics import *
from manipulation.motion.trajectories import *
from manipulation.constants import *
import time

from operator_utils.grasp_utils import solveTwoArmIKs
from operator_utils.grasp_utils import compute_two_arm_grasp, translate_point, \
    compute_Tee_at_given_Ttool

from utils import set_robot_config, trans_from_xytheta, compute_angle_to_be_set, \
    grab_obj, get_body_xytheta


def sample_angle_facing_given_transform(target_xy, robot_xy):
    """
    target_dirn = target_xy-robot_xy
    target_dirn = target_dirn/np.linalg.norm(target_dirn)
    if target_dirn[1] < 0:
      # rotation from x-axis, because that is the default rotation
      angle_to_be_set = -math.acos( np.dot(target_dirn,np.array(([1,0]))) )
    else:
      angle_to_be_set =  math.acos( np.dot(target_dirn,np.array(([1,0]))) )
    """
    angle_to_be_set = compute_angle_to_be_set(target_xy, robot_xy)
    dangle_in_rad = 30 * np.pi / 180.0  # random offset from the angle facing the object
    return angle_to_be_set + np.random.uniform(-dangle_in_rad, dangle_in_rad)


def sample_xy_locations(obj, radius):
    portion = np.random.uniform(0.4, 0.9)
    angle = np.random.uniform(0, 2 * PI)  # which angle?

    robot_base_pose = compute_robot_xy_given_ir_parameters(portion, angle, obj, radius)
    return robot_base_pose  # turn it into world ref frame


def sample_ir(obj, robot, env, region, n_iter=300):
    arm_len = PR2_ARM_LENGTH  # determined by spreading out the arm and measuring the dist from shoulder to ee
    # grasp_pos = Tgrasp[0:-1,3]
    obj_xy = get_point(obj)[:-1]
    robot_xy = get_point(robot)[:-1]
    dist_to_grasp = np.linalg.norm(robot_xy - obj_xy)

    n_samples = 1
    for _ in range(n_iter):
        robot_xy = sample_xy_locations(obj, arm_len)[:-1]
        angle = sample_angle_facing_given_transform(obj_xy, robot_xy)  # angle around z
        set_robot_config(np.r_[robot_xy, angle], robot)
        if (not env.CheckCollision(robot)) and (region.contains(robot.ComputeAABB())):
            return np.array([robot_xy[0], robot_xy[1], angle])
    return None


def sample_ir_multiple_regions(obj, robot, env, multiple_regions):
    arm_len = 0.9844  # determined by spreading out the arm and measuring the dist from shoulder to ee
    # grasp_pos = Tgrasp[0:-1,3]
    obj_xy = get_point(obj)[:-1]
    robot_xy = get_point(robot)[:-1]
    dist_to_grasp = np.linalg.norm(robot_xy - obj_xy)

    n_samples = 1
    for _ in range(300):
        robot_xy = sample_base_locations(arm_len, obj, env)[:-1]
        angle = sample_angle_facing_given_transform(obj_xy, robot_xy)  # angle around z
        set_robot_config(np.r_[robot_xy, angle], robot)
        if (not env.CheckCollision(robot)) and np.any([r.contains(robot.ComputeAABB()) for r in multiple_regions]):
            return np.array([robot_xy[0], robot_xy[1], angle])
    return None


def create_region(env, region_name, ((nx, px), (ny, py)), table_name, color=None):
    table_aabb = aabb_from_body(env.GetKinBody(table_name))
    return AARegion(region_name,
                    ((table_aabb.pos()[0] + nx * table_aabb.extents()[0],
                      table_aabb.pos()[0] + px * table_aabb.extents()[0]),
                     (table_aabb.pos()[1] + ny * table_aabb.extents()[1],
                      table_aabb.pos()[1] + py * table_aabb.extents()[1])),
                    table_aabb.pos()[2] + table_aabb.extents()[2] + REGION_Z_OFFSET, color=color)


def simulate_base_path(robot, path):
    for p in path:
        # set_config(robot, p, get_active_arm_indices(robot))
        set_robot_config(p, robot)
        sleep(0.001)


def randomly_place_in_region(env, body, region):
    if env.GetKinBody(get_name(body)) is None: env.Add(body)
    for i in range(1000):
        set_quat(body, quat_from_z_rot(uniform(0, 2 * PI)))
        aabb = aabb_from_body(body)
        cspace = region.cspace(aabb)
        if cspace is None: continue
        set_point(body, np.array([uniform(*cspace_range) for cspace_range in cspace] + [
            region.z + aabb.extents()[2] + BODY_PLACEMENT_Z_OFFSET]) - aabb.pos() + get_point(body))
        if not env.CheckCollision(body):
            return get_body_xytheta(body)
    return None


def gaussian_randomly_place_in_region(env, body, region, center, var):
    if env.GetKinBody(get_name(body)) is None:
        env.Add(body)

    for i in range(1000):
        xytheta = np.random.normal(center, var)
        set_obj_xytheta(xytheta, body)
        if not body_collision(env, body):
            return xytheta

    import pdb;pdb.set_trace()
    for i in range(1000):
        set_quat(body, quat_from_z_rot(uniform(0, 2 * PI)))
        aabb = aabb_from_body(body)
        cspace = region.cspace(aabb)
        if cspace is None: continue
        set_point(body, np.array([uniform(*cspace_range) for cspace_range in cspace] + [
            region.z + aabb.extents()[2] + BODY_PLACEMENT_Z_OFFSET]) - aabb.pos() + get_point(body))
        if not body_collision(env, body):
            return get_body_xytheta(body)
    return None


def randomly_place_in_region_need_to_be_fixed(env, obj, region, th=None):
    # todo fix this function
    min_x = region.box[0][0]
    max_x = region.box[0][1]
    min_y = region.box[1][0]
    max_y = region.box[1][1]

    aabb = aabb_from_body(obj)
    # try 1000 placements
    for _ in range(300):
        x = np.random.rand(1) * (max_x - min_x) + min_x
        y = np.random.rand(1) * (max_y - min_y) + min_y
        z = [region.z + aabb.extents()[2] + BODY_PLACEMENT_Z_OFFSET]
        xyz = np.array([x[0],y[0],z[0]]) - aabb.pos() + get_point(obj) # todo: recheck this function; I think it failed if obj was robot

        if th == None:
            th = np.random.rand(1) * 2 * np.pi
        set_point(obj, xyz)
        set_quat(obj, quat_from_angle_vector(th, np.array([0, 0, 1])))

        obj_quat = get_quat(obj)
        # refer to conversion between quaternions and euler angles on Wiki for the following eqn.
        assert (np.isclose(th, np.arccos(obj_quat[0]) * 2)
                or np.isclose(th, np.arccos(-obj_quat[0]) * 2))
        # print not env.CheckCollision(obj) and region.contains(obj.ComputeAABB())
        if not env.CheckCollision(obj) \
                and region.contains(obj.ComputeAABB()):
            return [x[0], y[0], th[0]]
    return None


def sample_grasp_parameters(n_smpls=1):
    theta = np.random.random(n_smpls) * (180 * PI / 180 - 45 * PI / 180) + 45 * PI / 180
    height_portion = np.random.random(n_smpls) * (1 - 0.5) + 0.5
    depth_portion = np.random.random(n_smpls) * (0.9 - 0.1) + 0.1
    """
    theta          = np.random.random(1)[0]*( 180*PI/180 - 0*PI/180) + 0*PI/180
    height_portion = np.random.random(1)[0]*(1-0)
    depth_portion  = np.random.random(1)[0]*(1-0)
    """
    return theta, height_portion, depth_portion


def sample_one_arm_grasp_parameters(n_smpls=1):
    theta = np.random.random(n_smpls) * (180 * PI / 180 - 45 * PI / 180) + 45 * PI / 180
    height_portion = np.random.random(n_smpls) * (1 - 0) + 0
    depth_portion = np.random.random(n_smpls) * (0.95 - 0.8) + 0.8
    # todo define depth portion based on the size of the gripper
    """
    theta          = np.random.random(1)[0]*( 180*PI/180 - 0*PI/180) + 0*PI/180
    height_portion = np.random.random(1)[0]*(1-0)
    depth_portion  = np.random.random(1)[0]*(1-0)
    """
    return theta, height_portion, depth_portion


def generate_pick_grasp_and_base_pose(generator, obj_shape, obj_point):
    scaled_c = generator.c_scaler.transform(np.array(obj_shape)[None, :])
    scaled_x = generator.generate(scaled_c, 1)
    x = generator.x_scaler.inverse_transform(scaled_x)
    grasp_params = x[0, :-3]
    rel_pick_base = x[0, -3:]

    th = rel_pick_base[-1]
    abs_pick_base = np.append(rel_pick_base[0:2] + obj_point[0:2], th)

    return grasp_params[0], grasp_params[1], grasp_params[2], abs_pick_base


def generate_obj_placement(generator, context_vec):
    # scaled_c = generator.c_scaler.transform(np.array(occ_vec*1)[None,:])
    scaled_c = generator.c_scaler.transform(context_vec)
    scaled_c = scaled_c.reshape(generator.dim_context)[None, :]
    scaled_x = generator.generate(scaled_c, 1)
    x = generator.x_scaler.inverse_transform(scaled_x)
    return x


def sample_pick(obj, robot, env, region):
    # takes the target object to be picked and returns:
    # - base pose to pick the object
    # - grasp parameters to pick the object
    # - grasp to pick the object
    # to which there is a valid path

    n_trials = 5000  # try 20 different samples
    for _ in range(n_trials):
        base_pose = sample_ir(obj, robot, env, region)
        if base_pose is None:
            continue
        set_robot_config(base_pose, robot)

        theta, height_portion, depth_portion = sample_grasp_parameters()

        grasps = compute_two_arm_grasp(depth_portion, height_portion, theta, obj, robot)
        g_config = solveTwoArmIKs(env, robot, obj, grasps)
        if g_config is None:
            continue

        for body in env.GetBodies():
            body.Enable(True)
        return base_pose, [theta, height_portion, depth_portion], g_config

    return None, None, None


def sample_placement(env, obj, robot, obj_region, robot_region):
    status = "Failed"
    path = None
    original_trans = robot.GetTransform()
    tried = []
    n_trials = 1  # try 5 different samples of placements
    T_r_wrt_o = np.dot(np.linalg.inv(obj.GetTransform()), robot.GetTransform())
    for _ in range(n_trials):  # at most n_trials number of path plans
        # print 'releasing obj'

        while True:
            sleep(GRAB_SLEEP_TIME)
            robot.Release(obj)
            robot.SetTransform(original_trans)

            # first sample collision-free object placement
            obj_xytheta = randomly_place_in_region(env, obj, obj_region)  # randomly place obj

            # compute the resulting robot transform
            new_T_robot = np.dot(obj.GetTransform(), T_r_wrt_o)
            robot.SetTransform(new_T_robot)
            robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
            robot_xytheta = robot.GetActiveDOFValues()
            set_robot_config(robot_xytheta, robot)
            new_T = robot.GetTransform()
            assert (np.all(np.isclose(new_T, new_T_robot)))
            if not (check_collision_except(robot, obj, env)) \
                    and (robot_region.contains(robot.ComputeAABB())):
                break

        sleep(GRAB_SLEEP_TIME)
        grab_obj(robot, obj)

        robot.SetTransform(original_trans)
        # print 'motion planning...'
        stime = time.time()
        for node_lim in [1000, 5000, np.inf]:
            print node_lim
            path, tpath, status = get_motion_plan(robot, robot_xytheta, env, maxiter=10, n_node_lim=node_lim)
            print 'done', status, time.time() - stime
            if status == 'HasSolution': break
        print 'done', status, time.time() - stime
        if status == "HasSolution":
            # print 'returning with solution'
            return obj_xytheta, robot_xytheta, path
    # print 'grabbing obj'
    sleep(GRAB_SLEEP_TIME)
    robot.Grab(obj)
    # print 'grabbed'
    return None, None, None


def sample_pick_using_gen(obj, obj_shape, robot, generator, env, region):
    # diable all bodies; imagine being able to pick anywhere
    for body in env.GetBodies():
        body.Enable(False)

    # enable the target and the robot
    obj.Enable(True)
    robot.Enable(True)

    original_trans = robot.GetTransform()
    n_trials = 100  # try 20 different samples
    for idx in range(n_trials):
        theta, height_portion, depth_portion, base_pose \
            = generate_pick_grasp_and_base_pose(generator, obj_shape, get_point(obj))
        set_robot_config(base_pose, robot)
        grasps = compute_two_arm_grasp(depth_portion, height_portion, theta, obj, robot)  # tool trans
        g_config = solveTwoArmIKs(env, robot, obj, grasps)  # turns obj collision off
        if g_config is None:
            continue
        for body in env.GetBodies():
            if body.GetName().find('_pt_') != -1: continue
            body.Enable(True)

        return base_pose, [theta, height_portion, depth_portion], g_config
    return None, None, None


def check_collision_except(body1, exception_body, env):
    return np.any([env.CheckCollision(body1, body) for body in env.GetBodies() if body != exception_body])


"""
def sample_placement_using_gen( env,obj,robot,p_samples,
                                obj_region,robot_region,
                                GRAB_SLEEP_TIME=0.05):
  # This script, with a given grasp of an object,
  # - finding colfree obj placement within 100 tries. no robot at this point
  # - checking colfree ik solution for robot; if not, trial is over
  # - if above two passes, colfree path finding; if not, trial is over
  status = "Failed"
  path= None
  original_trans = robot.GetTransform()
  obj_orig_trans = obj.GetTransform()
  tried = []
  n_trials = 1 # try 5 different samples of placements with a given grasp
  T_r_wrt_o = np.dot( np.linalg.inv( obj.GetTransform()), robot.GetTransform())
  for _ in range(n_trials):
    #print 'releasing obj'
    sleep(GRAB_SLEEP_TIME)
    robot.Release(obj)
    robot.SetTransform(original_trans)
    obj.SetTransform(obj_orig_trans)
    #print 'released'
    # get robot pose wrt obj

    # sample obj pose
    #print 'randmly placing obj'
    #obj_xytheta = randomly_place_in_region(env,obj,obj_region) # randomly place obj
    inCollision = True
    #for _ in range(500): # try hundred different collision free obj placement
      #obj_xytheta = generate_obj_placement(generator,context_vec) 
    np.random.shuffle(p_samples)
    for idx,obj_xytheta in enumerate(p_samples):
      if idx>100: break
      x = obj_xytheta[0]
      y = obj_xytheta[1]
      z = get_point(obj)[-1]
      set_point(obj,[x,y,z])
      th=obj_xytheta[2]
      set_quat( obj, quat_from_z_rot(th) )

      new_T_robot = np.dot( obj.GetTransform(),T_r_wrt_o) 
      robot.SetTransform(new_T_robot)

      inCollision = (check_collision_except(obj,robot,env))\
                    or (check_collision_except(robot,obj,env))
      inRegion = (robot_region.contains(robot.ComputeAABB())) and\
                 (obj_region.contains(obj.ComputeAABB()))
      if (not inCollision) and inRegion:
        break
    if inCollision or not(inRegion):
      print 'obj in collision'
      break # if you tried all p samples and ran out, get new pick
    
    print 'obj in a feasible place'
    # compute the resulting robot transform
    sleep(GRAB_SLEEP_TIME)
    robot.Grab(obj)
    robot.SetTransform(new_T_robot)
    robot.SetActiveDOFs([],DOFAffine.X|DOFAffine.Y|DOFAffine.RotationAxis,[0,0,1])
    robot_xytheta=robot.GetActiveDOFValues()
    robot.SetTransform(original_trans)
    stime=time.time()
    print 'motion planning...'
    for node_lim in [1000,5000,np.inf]:
      print node_lim
      path,tpath,status = get_motion_plan(robot,\
                  robot_xytheta,env,maxiter=10,n_node_lim=node_lim)
      if path=='collision':
      #  import pdb;pdb.set_trace()
        pass
      print 'done',status,time.time()-stime
      if status == "HasSolution":  
        print 'returning with solution',tpath
        robot.SetTransform(new_T_robot)
        return obj_xytheta,robot_xytheta,path
      else:
        print 'motion planning failed',tpath
  sleep(GRAB_SLEEP_TIME)
  robot.Grab(obj)
  robot.SetTransform(original_trans)
  print "Returnining no solution" 
  return None,None,None
"""


def sample_placement_using_body_gen(env, obj, robot, p_samples,
                                    obj_region, robot_region,
                                    GRAB_SLEEP_TIME=0.05):
    # This script, with a given grasp of an object,
    # - finding colfree obj placement within 100 tries. no robot at this point
    # - checking colfree ik solution for robot; if not, trial is over
    # - if above two passes, colfree path finding; if not, trial is over
    status = "Failed"
    path = None
    original_trans = robot.GetTransform()
    obj_orig_trans = obj.GetTransform()
    tried = []
    n_trials = 1  # try 5 different samples of placements with a given grasp
    T_r_wrt_o = np.dot(np.linalg.inv(obj.GetTransform()), robot.GetTransform())
    for _ in range(n_trials):
        robot.SetTransform(original_trans)

        # sample obj pose
        inCollision = True
        np.random.shuffle(p_samples)
        for idx, xytheta in enumerate(p_samples):
            if idx > 100: break
            set_robot_config(xytheta, robot)

            inCollision = (check_collision_except(obj, robot, env)) \
                          or (check_collision_except(robot, obj, env))
            inRegion = (robot_region.contains(robot.ComputeAABB())) and \
                       (obj_region.contains(obj.ComputeAABB()))
            if (not inCollision) and inRegion:
                break
        if inCollision or not (inRegion):
            print 'obj in collision'
            break  # if you tried all p samples and ran out, get new pick
        print 'robot in a feasible place'

        # compute the resulting robot transform
        robot_xytheta = xytheta
        robot.SetTransform(original_trans)
        stime = time.time()
        print 'motion planning...'
        for node_lim in [1000, 5000, np.inf]:
            print node_lim
            path, tpath, status = get_motion_plan(robot, \
                                                  robot_xytheta, env, maxiter=10, n_node_lim=node_lim)
            if path == 'collision':
                #  import pdb;pdb.set_trace()
                pass
            print 'done', status, time.time() - stime
            if status == "HasSolution":
                print 'returning with solution', tpath
                set_robot_config(xytheta, robot)
                return None, robot_xytheta, path
            else:
                print 'motion planning failed', tpath
    robot.SetTransform(original_trans)
    print "Returnining no solution"
    return None, None, None


def trimesh_body(env, radius, dz, name=None, color=None):
    geom_info = KinBody.Link.GeometryInfo()
    geom_info._type = KinBody.Link.GeomType.Cylinder
    geom_info._t[
        3, 3] = dz / 2  # Local transformation of the geom primitive with respect to the link's coordinate system.
    geom_info._vGeomData = [radius, dz]
    # boxes - first 3 values are extents
    # sphere - radius
    # cylinder - first 2 values are radius and height
    # trimesh - none
    geom_info._bVisible = True
    geom_info._fTransparency = 0.0  # value from 0-1 for the transparency of the rendered object, 0 is opaque
    if color is not None: geom_info._vDiffuseColor = color
    body = RaveCreateKinBody(env, '')
    body.InitFromGeometries([geom_info])
    if name is not None: set_name(body, name)
    return body
