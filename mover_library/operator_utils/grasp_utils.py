
from time import time
import numpy as np


from manipulation.primitives.transforms import trans_from_base_values, set_pose, set_quat, \
    point_from_pose, axis_angle_from_rot, \
    rot_from_quat, quat_from_pose, quat_from_z_rot, \
    get_pose, base_values_from_pose, \
    pose_from_base_values, get_point

from manipulation.primitives.inverse_kinematics import inverse_kinematics_helper
from manipulation.bodies.bodies import *
from manipulation.primitives.transforms import *
from manipulation.primitives.savers import *

import time


def translate_point(target_transform, point):
    if len(point) == 3:
        point.concatenate(np.concatenate([point, [1]]))
    elif len(point) != 4:
        print 'Invalid dimension'
        return
    transformed_point = trans_dot(target_transform, point)  # equation 2.23 in Murray
    return transformed_point


def tool_wrt_world(roll, pitch, yaw, tool_point):
    desired_roll = quat_from_angle_vector(roll, np.array([1, 0, 0]))
    desired_pitch = quat_from_angle_vector(pitch, np.array([0, 1, 0]))
    desired_yaw = quat_from_angle_vector(yaw, np.array([0, 0, 1]))
    tool_rot_wrt_w = quat_dot(desired_yaw, desired_pitch, desired_roll)
    desired_tool_wrt_w = trans_from_quat_point(tool_rot_wrt_w, tool_point)
    return desired_tool_wrt_w


def compute_tool_trans_wrt_obj_trans(tool_trans_wrt_world, object_trans_wrt_world):
    # tool_trans_wrt_world - tool trans in world ref frame
    # obj_trans_wrt_world  - obj trans in world ref frame
    # output:
    # solves T_world * T_obj = O_world,
    # for T_obj and returns it.
    # T_world and T_obj are tool transform wrt world and object respectively
    # and O_world is the obj trans wrt world

    # I think this computing transform of object wrt tool transform?
    return np.linalg.solve(object_trans_wrt_world, tool_trans_wrt_world)


def compute_Tee_at_given_Ttool(tool_trans_wrt_world, tool_trans_wrt_ee):
    return np.dot(tool_trans_wrt_world, np.linalg.inv(tool_trans_wrt_ee))


def compute_grasp_global_transform(gtrans, obj):
    return np.dot(get_trans(obj), gtrans)


def compute_one_arm_grasp(depth_portion, height_portion, theta, obj, robot):
    # Compute grasps at four different directions at 5 different heights, total of four different grasps
    # grasp = global tool transform

    PR2_GRIPPER_LENGTH = 0.025
    T_tools_wrt_obj = []
    manip = robot.GetActiveManipulator()
    ee = manip.GetEndEffector()
    with obj:
        obj.SetTransform(np.eye(4))
        aabb = obj.ComputeAABB()
        x_extent = aabb.extents()[0]
        y_extent = aabb.extents()[1]
        z_extent = aabb.extents()[2]

        for yaw in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            roll = 0
            pitch = theta #np.pi / 2

            if yaw == np.pi / 2:
                grasp_axis = np.array([0, 1, 0])
                obj_extent_in_grasp_dir = y_extent
            elif yaw == 3 * np.pi / 2:
                grasp_axis = np.array([0, -1, 0])
                obj_extent_in_grasp_dir = y_extent
            elif yaw == 0:
                grasp_axis = np.array([1, 0, 0])
                obj_extent_in_grasp_dir = x_extent
            elif yaw == np.pi:
                grasp_axis = np.array([-1, 0, 0])
                obj_extent_in_grasp_dir = x_extent

            # I want this to be:
            # out of reach if depth_portion is 0
            # in full contact with palm if depth portion is 1
            grasp_depth = grasp_axis * (-PR2_GRIPPER_LENGTH + depth_portion*2*PR2_GRIPPER_LENGTH) #obj_extent_in_grasp_dir * (-depth_portion*PR2_GRIPPER_LENGTH)
            obj_center_xyz = aabb.pos()
            tool_point = obj_center_xyz + grasp_depth

            grasp_height_portion = height_portion
            grasp_height = np.array([0, 0, 1]) * (-z_extent + 2 * z_extent * grasp_height_portion)

            T_tool_wrt_obj = tool_wrt_world(roll, pitch, yaw, tool_point + grasp_height)
            #desired_ee_world = compute_Tee_at_given_Ttool(T_tool_wrt_obj, manip.GetLocalToolTransform())
            #visualize_grasp(manip, desired_ee_world)
            T_tools_wrt_obj.append(T_tool_wrt_obj)

    grasps = [compute_grasp_global_transform(T, obj) for T in T_tools_wrt_obj]
    return grasps


def compute_two_arm_grasp(depth_portion, height_portion, theta, obj, robot):
    # This function computes four different two-arm grasps for box shaped objects at each side
    # A grasp is a transform of the robot's end-effector in world reference frame
    # Grasp is parametrized by four parameters:
    #   theta - controls the angle with respect to the grasp axis, ranges from 0 to PI
    #   depth_portion - determines how deep grasp axis is on the obj
    #   height_portion - determines how how high the grasp axis is on the obj
    # Grasp axis is where the left and right arm tools align

    grasp_list = []

    with obj:
        obj.SetTransform(np.eye(4))

        aabb = obj.ComputeAABB()
        x_extent = aabb.extents()[0]
        y_extent = aabb.extents()[1]
        z_extent = aabb.extents()[2]

        if obj.GetName().find('tobj') != -1:
            aabb = obj.GetLinks()[0].ComputeAABB()
            x_extent = aabb.extents()[0]
            y_extent = aabb.extents()[1]
            z_extent = aabb.extents()[2]

        # yaw list is the relative orientation of grasp wrt obj
        yaw_list = [0, PI / 2, PI, 3 * PI / 2]  # iterate through four sides
        grasps = []
        for yaw in yaw_list:
            if yaw == PI / 2 or yaw == 3 * PI / 2:
                roll = theta
                pitch = 0
                # these directions are so adhoc, computed using trial-and-error
                if yaw == PI / 2:
                    grasp_axis = np.array([0, 1, 0])
                    non_grasp_axis = np.array([-1, 0, 0])
                else:
                    grasp_axis = np.array([0, -1, 0])
                    non_grasp_axis = np.array([1, 0, 0])
                extent = y_extent
                depth = x_extent
            else:
                roll = theta
                pitch = 0
                # these directions are so adhoc, computed using trial-and-error
                if yaw == 0:
                    grasp_axis = np.array([1, 0, 0])
                    non_grasp_axis = np.array([0, 1, 0])
                else:
                    grasp_axis = np.array([-1, 0, 0])
                    non_grasp_axis = np.array([0, -1, 0])
                extent = x_extent
                depth = y_extent

            # compute the grasp point on the object surface
            grasp_width = grasp_axis * (extent + 0.045)
            grasp_depth = non_grasp_axis * (-depth + 2 * depth * depth_portion)
            grasp_height = np.array([0, 0, 1]) * (z_extent - 2 * z_extent * height_portion)

            # grasp point on the obj for the right arm
            tool_point = aabb.pos() - grasp_width - grasp_depth - grasp_height
            robot.SetActiveManipulator('rightarm_torso')
            manip = robot.GetManipulator('rightarm_torso');
            rightarm_tool = tool_wrt_world(roll, pitch, yaw, tool_point)
            # manip.GetEndEffector().SetTransform(rightarm_tool)
            # visualize_grasp(manip, rightarm_tool)

            # grasp point on the obj for the left arm
            tool_point = aabb.pos() + grasp_width - grasp_depth - grasp_height
            robot.SetActiveManipulator('leftarm')
            manip = robot.GetManipulator('leftarm');
            leftarm_tool = tool_wrt_world(roll, pitch, yaw, tool_point)
            # manip.GetEndEffector().SetTransform(leftarm_tool)

            grasps.append([leftarm_tool, rightarm_tool, yaw])
            # robot.GetEnv().UpdatePublishedBodies()

    grasp_list = [[compute_grasp_global_transform(g[0], obj), \
                   compute_grasp_global_transform(g[1], obj), \
                   g[2]]
                  for g in grasps]
    return grasp_list


def solveIK(env, robot, tool_trans_wrt_world):
    g_config = inverse_kinematics_helper(env, robot, tool_trans_wrt_world)
    return g_config


def solveIKs(env, robot, grasps):
    # returns a feasible IK solution given multiple grasps
    for g in grasps:
        g_config = solveIK(env, robot, g)
        if g_config is not None:
            return g_config, g
    return None, None


def visualize_grasp(manip, Tworld):
    gripper = manip.GetEndEffector()
    manip.GetRobot().GetEnv().UpdatePublishedBodies()
    gripper.SetTransform(Tworld)

    """
    Tleft_ee = compute_Tee_at_given_Ttool(g_left,\
                                           leftarm_manip.GetLocalToolTransform())
    Tright_ee = compute_Tee_at_given_Ttool(g_right,\
                                           rightarm_manip.GetLocalToolTransform())
    """

def solveTwoArmIKs(env, robot, obj, grasps):
    leftarm_manip = robot.GetManipulator('leftarm')
    rightarm_manip = robot.GetManipulator('rightarm')
    rightarm_torso_manip = robot.GetManipulator('rightarm_torso')
    arm_len = 0.9844  # determined by spreading the arm and measuring the dist from shoulder to ee

    # a grasp consists of:
    # g[0] = left grasp
    # g[1] = right grasp
    # g[2] = grasp wrt obj; used to filter out robot poses that are not facing the same direction
    for g in grasps:
        # first check if g within reach
        g_left = g[0]
        g_right = g[1]
        yaw_wrt_obj = g[2]
        Tleft_ee = compute_Tee_at_given_Ttool(g_left, \
                                              leftarm_manip.GetLocalToolTransform())
        Tright_ee = compute_Tee_at_given_Ttool(g_right, \
                                               rightarm_manip.GetLocalToolTransform())

        left_ee_xy = point_from_trans(Tleft_ee)[:-1]
        right_ee_xy = point_from_trans(Tright_ee)[:-1]
        mid_grasp_xy = (left_ee_xy + right_ee_xy) / 2
        robot_xy = get_point(robot)[:-1]
        obj_xy = get_point(obj)[:-1]

        # filter the trivial conditions IK wont be found
        # condition 1:
        # robot and the grasp must face the same direction
        # where is robot facing?
        r_quat = get_quat(robot)
        o_quat = get_quat(obj)
        # refer to wiki on Conversion between Euler and Quaternion for these eqns
        r_z_rot = np.arccos(r_quat[0]) * 2 if r_quat[-1] >= 0 else np.arccos(-r_quat[0]) * 2
        o_z_rot = np.arccos(o_quat[0]) * 2 if o_quat[-1] >= 0 else np.arccos(-o_quat[0]) * 2
        r_z_rot *= 180 / PI;
        o_z_rot *= 180 / PI
        if r_z_rot < 0: r_z_rot + 360
        if o_z_rot < 0: o_z_rot + 360
        angle_diff = r_z_rot - o_z_rot
        if angle_diff < 0: angle_diff += 360
        # ugh, look at the notes to figure out how I did the thing below
        if (angle_diff < 45 or (angle_diff <= 360 and angle_diff >= 315) and (yaw_wrt_obj != PI / 2)) \
                or (angle_diff < 135 and angle_diff >= 45 and yaw_wrt_obj != PI) \
                or (angle_diff < 225 and angle_diff >= 135 and yaw_wrt_obj != 3 * PI / 2) \
                or (angle_diff < 310 and angle_diff >= 225 and yaw_wrt_obj != 0):
            continue

        # condition 2: ee loc must be within reach
        right_ee_dist = np.linalg.norm(robot_xy - right_ee_xy)
        left_ee_dist = np.linalg.norm(robot_xy - left_ee_xy)
        if right_ee_dist > arm_len * 0.75 or left_ee_dist > arm_len * 0.75:
            continue

        # checking right arm ik solution feasibility
        obj.Enable(False)
        right_g_config = rightarm_torso_manip.FindIKSolution(g_right, 0)

        # rightarm_torso_manip.GetEndEffector().SetTransform(Tright_ee)
        if right_g_config is None:
            obj.Enable(True)
            continue

        # turning checkenvcollision option for FindIKSolution seems take excessive amt of time
        with robot:
            set_config(robot, right_g_config, rightarm_torso_manip.GetArmIndices())
            if env.CheckCollision(robot):
                right_g_config = None

        # checking left arm ik solution feasibility
        st = time.time()
        obj.Enable(False)
        left_g_config = leftarm_manip.FindIKSolution(g_left, 0)
        with robot:
            set_config(robot, left_g_config, leftarm_manip.GetArmIndices())
            if env.CheckCollision(robot):
                left_g_config = None

        if left_g_config is None:
            obj.Enable(True)
            continue

        obj.Enable(True)
        if not (left_g_config is None) and not (right_g_config is None):
            obj.Enable(True)
            return [left_g_config, right_g_config]
    return None


"""
def IK_helper(env, robot, grasp_transform, manip):
  with robot:
    robot.SetActiveDOFs(manip.GetArmIndices())
    with collision_saver(env, openravepy_int.CollisionOptions.ActiveDOFs):
      #print get_manipulator(robot).GetIkSolver()
      config = manip.FindIKSolution(grasp_transform, 0) 
      if config is None: return None
      set_config(robot, config, manip.GetArmIndices())
      if env.CheckCollision(robot) or robot.CheckSelfCollision(): return None
      return config
"""
