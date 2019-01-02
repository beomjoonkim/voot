import numpy as np
import sys
import copy
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../mover_library/')
from conveyor_belt_problem import two_tables_through_door
from utils import compute_occ_vec, set_robot_config, remove_drawn_configs, \
    draw_configs, clean_pose_data, draw_robot_at_conf, \
    pick_obj, place_obj, check_collision_except, release_obj

from manipulation.primitives.savers import DynamicEnvironmentStateSaver
from motion_planner import collision_fn, base_extend_fn, base_sample_fn, base_distance_fn, smooth_path, rrt_connect


from utils import *
from operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp

from openravepy import *
import time


class DynamicEnvironmentStateSaverWithCurrObj(DynamicEnvironmentStateSaver):
    def __init__(self, env, placements, curr_obj, is_pick_node):
        DynamicEnvironmentStateSaver.__init__(self, env)
        self.curr_obj = curr_obj
        self.is_pick_node = is_pick_node
        self.placements = placements


class ConveyorBelt:
    def __init__(self, v=False):
        self.env = Environment()
        self.problem = two_tables_through_door(self.env)
        self.objects = self.problem['objects']
        self.robot = self.env.GetRobots()[0]
        self.init_base_conf = np.array([0, 1.05, 0])
        self.all_region = self.problem['all_region']
        self.loading_region = self.problem['loading_region']
        #self.key_configs = pickle.load(open('./key_configs/key_configs.p', 'r'))
        self.v = v
        self.infeasible_reward = -2

        if self.v:
            self.env.SetViewer('qtcoin')
        self.curr_obj = self.objects[0]
        self.curr_state = self.get_state()
        self.initial_placements = []
        self.placements = []
        self.init_saver = DynamicEnvironmentStateSaverWithCurrObj(self.env, self.get_placements(), self.curr_obj, False)
        self.is_init_pick_node = True
        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    def restore(self, state_saver):
        curr_obj = state_saver.curr_obj
        is_pick_time = state_saver.is_pick_node
        if not is_pick_time:
            grab_obj(self.robot, curr_obj)
        else:
            if len(self.robot.GetGrabbed()) > 0:
                release_obj(self.robot, self.robot.GetGrabbed()[0])
        state_saver.Restore()
        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    def reset_to_init_state(self):
        self.init_saver.Restore()
        self.curr_state = self.get_state()
        self.placements = copy.deepcopy(self.initial_placements)
        self.curr_obj = self.objects[len(self.placements)]
        if not self.is_init_pick_node:
            grab_obj(self.robot, self.curr_obj)
        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    def enable_movable_objects(self):
        for obj in self.objects:
            obj.Enable(True)

    def disable_movable_objects(self):
        for obj in self.objects:
            obj.Enable(False)

    def set_init_state(self, saver):
        self.init_saver = saver
        self.initial_placements = copy.deepcopy(saver.placements)
        self.is_init_pick_node = saver.is_pick_node

    def get_state_saver(self):
        return DynamicEnvironmentStateSaverWithCurrObj(self.env, self.get_placements(), self.curr_obj,
                                                       self.is_pick_time())

    def get_curr_object(self):
        return self.curr_obj

    def get_placements(self):
        return copy.deepcopy(self.placements)

    def get_state(self):
        return 1
        # our state is represented with a key configuration collision vector
        #c_data = compute_occ_vec(self.key_configs, self.robot, self.env)[None, :] * 1
        #scaled_c = convert_collision_vec_to_one_hot(c_data)
        #c_data = np.tile(scaled_c, (1, 1, 1))
        #c_data = c_data[:, :, :, None]
        #return c_data

    def apply_pick_action(self, action, obj=None):
        leftarm_manip = self.robot.GetManipulator('leftarm')
        rightarm_torso_manip = self.robot.GetManipulator('rightarm_torso')

        if obj is None:
            obj_to_pick = self.curr_obj
        else:
            obj_to_pick = obj
        pick_base_pose, grasp_params = action
        set_robot_config(pick_base_pose, self.robot)
        grasps = compute_two_arm_grasp(depth_portion=grasp_params[2],
                                       height_portion=grasp_params[1],
                                       theta=grasp_params[0],
                                       obj=obj_to_pick,
                                       robot=self.robot)
        g_config = solveTwoArmIKs(self.env, self.robot, obj_to_pick, grasps)
        assert g_config is not None

        pick_obj(obj_to_pick, self.robot, g_config, leftarm_manip, rightarm_torso_manip)
        set_robot_config(self.init_base_conf, self.robot)
        curr_state = self.get_state()
        reward = 0
        return curr_state, reward, g_config

    def update_next_obj_to_pick(self, place_action):
        self.placements.append(place_action)
        if len(self.placements) < len(self.objects):
            self.curr_obj = self.objects[len(self.placements)]  # update the next object to be picked

    def apply_place_action(self, action, do_check_reachability=True):
        # todo should this function tell you that it is a terminal state?
        robot = self.robot
        path, is_action_feasible = self.check_action_feasible(action, do_check_reachability)
        if is_action_feasible:
            place_robot_pose = action[0, :]
            self.place_object(place_robot_pose)
            self.curr_state = self.get_state()
            self.update_next_obj_to_pick(action)
            reward = 1

            is_goal_state = len(self.placements) == len(self.objects)
            if is_goal_state:
                return self.curr_state, reward, path
            return self.curr_state, reward, path
        else:
            return self.curr_state, self.infeasible_reward, path

    def place_object(self, place_base_pose, object=None):
        if object is None:
            obj_to_place = self.robot.GetGrabbed()[0]
        else:
            obj_to_place = object

        robot = self.robot
        leftarm_manip = robot.GetManipulator('leftarm')
        rightarm_manip = robot.GetManipulator('rightarm')

        set_robot_config(place_base_pose, robot)
        place_obj(obj_to_place, robot, leftarm_manip, rightarm_manip)

    def check_reachability(self, goal):
        if self.v:
            draw_robot_at_conf(goal, 0, 'goal', self.robot, self.env)

        path, status = self.get_motion_plan(goal)

        if self.v:
            remove_drawn_configs('goal', self.env)

        return path, status

    def check_action_feasible(self, action, do_check_reachability=True):
        action = action.reshape((1, action.shape[-1]))
        place_robot_pose = action[0, 0:3]

        if not self.is_collision_at_base_pose(place_robot_pose):
            if do_check_reachability:
                path, status = self.check_reachability(place_robot_pose)
                if status == "HasSolution":
                    return path, True
                else:
                    return None, False
            else:
                return None, True
        else:
            return None, False

    def visualize_placements(self, pi):
        # used for debugging purposes
        cvec = self.get_state()
        is_unif_policy = pi.__module__.find("Unif") != -1
        if is_unif_policy:
            samples = pi.predict(self.curr_obj, n_samples=10)
        else:
            samples = pi.predict(cvec, n_samples=10)

        draw_robot_base_configs(samples, self.robot, self.env)
        raw_input("Continue?")
        remove_drawn_configs('bconf', self.env)

    def is_collision_at_base_pose(self, base_pose, obj=None):
        robot = self.robot
        env = self.env
        if obj is None:
            obj_holding = self.curr_obj
        else:
            obj_holding = obj
        with robot:
            set_robot_config(base_pose, robot)
            in_collision = (check_collision_except(obj_holding, robot, env)) \
                           or (check_collision_except(robot, obj_holding, env))
            in_region = (self.all_region.contains(robot.ComputeAABB())) and \
                        (self.loading_region.contains(obj_holding.ComputeAABB()))
        if in_collision or (not in_region):
            return True
        return False

    def is_pick_time(self):
        return len(self.robot.GetGrabbed()) == 0

    def remove_all_obstacles(self):
        for body in self.env.GetBodies():
            if body.GetName().find('obst') != -1:
                self.env.Remove(body)

    def get_motion_plan(self, goal):
        d_fn = base_distance_fn(self.robot, x_extents=2.51, y_extents=2.51)
        s_fn = base_sample_fn(self.robot, x_extents=2.51, y_extents=2.51)
        e_fn = base_extend_fn(self.robot)
        c_fn = collision_fn(self.env, self.robot)
        q_init = self.robot.GetActiveDOFValues()

        n_iterations = [20, 50, 100, 500, 1000]
        print "Path planning..."
        stime = time.time()
        for n_iter in n_iterations:
            path = rrt_connect(q_init, goal, d_fn, s_fn, e_fn, c_fn, iterations=n_iter)
            if path is not None:
                path = smooth_path(path, e_fn, c_fn)
                print "Path Found, took %.2f"%(time.time()-stime)
                return path, "HasSolution"

        print "Path not found, took %.2f"%(time.time()-stime)
        return None, 'NoPath'


