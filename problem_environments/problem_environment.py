import copy
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../mover_library/')
from openravepy import DOFAffine, Environment

from utils import grab_obj, release_obj, set_robot_config,  check_collision_except, \
    draw_robot_at_conf, remove_drawn_configs, visualize_path, open_gripper, set_active_dof_conf, \
    two_arm_pick_object, two_arm_place_object, one_arm_pick_object, one_arm_place_object

from motion_planner import collision_fn, base_extend_fn, base_sample_fn, base_distance_fn, extend_fn, distance_fn, \
    sample_fn, smooth_path, rrt_connect, arm_base_sample_fn, arm_base_distance_fn, arm_base_extend_fn
import time


class ProblemEnvironment:
    def __init__(self):
        self.env = Environment()
        self.initial_placements = []
        self.placements = []
        self.robot = None
        self.objects = None
        self.curr_state = None
        self.curr_obj = None
        self.init_saver = None
        self.init_which_opreator = None
        self.v = False
        self.robot_region = None
        self.obj_region = None
        self.objs_to_move = None
        self.problem_config = None
        self.init_objs_to_move = None
        self.optimal_score = None
        self.name = None

        self.is_solving_packing = False
        self.is_solving_namo = False
        self.is_solving_fetching = False

        self.high_level_planner = None
        self.namo_planner = None
        self.fetch_planner = None
        self.env.StopSimulation()

    def make_config_from_op_instance(self, op_instance):
        if op_instance['operator'].find('one_arm')!=-1:
            g_config = op_instance['action']['g_config']
            base_pose = op_instance['action']['base_pose']
            config = np.hstack([g_config, base_pose.squeeze()])
        else:
            config = op_instance['action']['base_pose']

        return config.squeeze()

    def reset_to_init_state_stripstream(self):
        self.init_saver.Restore()
        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    def enable_movable_objects(self):
        for obj in self.objects:
            obj.Enable(True)

    def disable_movable_objects(self):
        for obj in self.objects:
            obj.Enable(False)

    def get_curr_object(self):
        return self.curr_obj

    def get_placements(self):
        return copy.deepcopy(self.placements)

    def get_state(self):
        return 1

    def is_pick_time(self):
        return len(self.robot.GetGrabbed()) == 0

    def check_action_feasible(self, action, do_check_reachability=True, region_name=None):
        action = action.reshape((1, action.shape[-1]))
        place_robot_pose = action[0, 0:3]

        if not self.is_collision_at_base_pose(place_robot_pose):
            if do_check_reachability:
                # define the region to stay in?
                path, status = self.check_reachability(place_robot_pose, region_name)
                if status == "HasSolution":
                    return path, True
                else:
                    return None, False
            else:
                return None, True
        else:
            return None, False

    def is_collision_at_base_pose(self, base_pose, obj=None):
        robot = self.robot
        env = self.env
        if obj is None:
            obj_holding = self.curr_obj
        else:
            obj_holding = obj
        with robot:
            set_robot_config(base_pose, robot)
            in_collision = check_collision_except(obj_holding, env)
        if in_collision:
            return True
        return False

    def is_in_region_at_base_pose(self, base_pose, obj, robot_region, obj_region):
        robot = self.robot
        if obj is None:
            obj_holding = self.curr_obj
        else:
            obj_holding = obj

        with robot:
            set_robot_config(base_pose, robot)
            in_region = (robot_region.contains(robot.ComputeAABB())) and \
                        (obj_region.contains(obj_holding.ComputeAABB()))
        return in_region

    def get_motion_plan(self, q_init, goal, d_fn, s_fn, e_fn, c_fn, n_iterations):
        #if self.env.GetViewer() is not None: #and not self.is_solving_ramo:
        #    draw_robot_at_conf(goal, 0.5, 'goal', self.robot, self.env)

        stime = time.time()
        for n_iter in n_iterations:
            path = rrt_connect(q_init, goal, d_fn, s_fn, e_fn, c_fn, iterations=n_iter)
            if path is not None:
                path = smooth_path(path, e_fn, c_fn)
                #print "Path Found, took %.2f"%(time.time()-stime)

                #if self.env.GetViewer() is not None:
                #    remove_drawn_configs('goal', self.env)
                return path, "HasSolution"

        #if self.env.GetViewer() is not None: #and not self.is_solving_ramo:
        #    remove_drawn_configs('goal', self.env)

        #print "Path not found, took %.2f"%(time.time()-stime)
        return None, 'NoPath'

    def get_arm_base_motion_plan(self, goal, region_name=None, manip_name=None):
        if region_name is None:
            d_fn = arm_base_distance_fn(self.robot, 2.51, 2.51)
            s_fn = arm_base_sample_fn(self.robot, 2.51, 2.51)
        else:
            region_x = self.problem_config[region_name+'_xy'][0]
            region_y = self.problem_config[region_name+'_xy'][1]
            region_x_extents = self.problem_config[region_name+'_extents'][0]
            region_y_extents = self.problem_config[region_name+'_extents'][1]
            d_fn = arm_base_distance_fn(self.robot, region_x_extents, region_y_extents)
            s_fn = arm_base_sample_fn(self.robot, region_x_extents, region_y_extents, region_x, region_y)
        e_fn = arm_base_extend_fn(self.robot)
        c_fn = collision_fn(self.env, self.robot)

        if manip_name is not None:
            manip = self.robot.GetManipulator(manip_name)
        else:
            manip = self.robot.GetManipulator('rightarm_torso')
        self.robot.SetActiveDOFs(manip.GetArmIndices(), DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

        q_init = self.robot.GetActiveDOFValues()
        n_iterations = [20, 50, 100, 500, 1000]
        print 'Arm base motion planning...'
        path, status = self.get_motion_plan(q_init, goal, d_fn, s_fn, e_fn, c_fn, n_iterations)
        return path, status

    def get_base_motion_plan(self, goal, region_name=None, n_iterations=None):
        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
        if region_name is None:
            assert self.name == 'convbelt'
            d_fn = base_distance_fn(self.robot, x_extents=3.51, y_extents=2.51)
            s_fn = base_sample_fn(self.robot, x_extents=3.51, y_extents=2.51) # set the x and y
        else:
            # todo: combined regions
            if region_name == 'bridge_region':
                region_name = 'entire_region'
            region_x = self.problem_config[region_name+'_xy'][0]
            region_y = self.problem_config[region_name+'_xy'][1]
            region_x_extents = self.problem_config[region_name+'_extents'][0]
            region_y_extents = self.problem_config[region_name+'_extents'][1]
            d_fn = base_distance_fn(self.robot, x_extents=region_x_extents, y_extents=region_y_extents)
            s_fn = base_sample_fn(self.robot, x_extents=region_x_extents, y_extents=region_y_extents, x=region_x, y=region_y)
        e_fn = base_extend_fn(self.robot)
        c_fn = collision_fn(self.env, self.robot)
        q_init = self.robot.GetActiveDOFValues()

        if n_iterations is None:
            if self.name == 'convbelt':
                #n_iterations = [20, 50, 100, 500, 3000]
                n_iterations = [20, 50, 100, 500] #700]
            else:
                n_iterations = [20, 50, 100, 500, 1000]

        print "Base motion planning..."
        path, status = self.get_motion_plan(q_init, goal, d_fn, s_fn, e_fn, c_fn, n_iterations)
        print "Status,",status
        return path, status

    def get_arm_motion_plan(self, goal, manip_name=None):
        if manip_name is not None:
            manip = self.robot.GetManipulator(manip_name)
        else:
            manip = self.robot.GetManipulator('rightarm_torso')
        self.robot.SetActiveDOFs(manip.GetArmIndices())

        d_fn = distance_fn(self.robot)
        s_fn = sample_fn(self.robot)
        e_fn = extend_fn(self.robot)
        c_fn = collision_fn(self.env, self.robot, check_self=True)

        q_init = self.robot.GetActiveDOFValues()
        n_iterations = [20, 50, 100, 500, 1000]
        print "Arm motion planning..."
        path, status = self.get_motion_plan(q_init, goal, d_fn, s_fn, e_fn, c_fn, n_iterations)
        return path, status

    def apply_constraint_removal_plan(self, plan):
        print "Applying constraint removal plan"
        for plan_step in plan:
            obj = self.env.GetKinBody(plan_step['obj_name'])
            action = plan_step['action']
            operator = plan_step['operator']
            if operator == 'two_arm_pick':
                full_body_pick_config = plan_step['path'][0]
                pick_base_pose = action[0]
                set_robot_config(pick_base_pose, self.robot)
                self.robot.SetDOFValues(full_body_pick_config)
                grab_obj(self.robot, obj)
            elif operator == 'two_arm_place':
                place_base_pose = action
                self.place_object(place_base_pose, obj)

    def visualize_crp(self, crp):
        for plan_step in crp:
            obj = self.env.GetKinBody(plan_step['obj_name'])
            action = plan_step['action']
            operator = plan_step['operator']
            if operator == 'two_arm_pick':
                full_body_pick_config = plan_step['path'][0]
                pick_base_pose = action[0]
                pick_path = plan_step['path'][1]
                self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
                visualize_path(self.robot, pick_path)
                set_robot_config(pick_base_pose, self.robot)
                self.robot.SetDOFValues(full_body_pick_config)
                grab_obj(self.robot, obj)
            elif operator == 'two_arm_place':
                place_base_pose = action
                place_path = plan_step['path']
                visualize_path(self.robot, place_path)
                self.place_object(place_base_pose, obj)

    def apply_plan(self, plan):
        for plan_step in plan:
            obj = self.env.GetKinBody(plan_step['obj_name'])
            action = plan_step['action']
            operator = plan_step['operator']
            if operator == 'two_arm_pick':
                two_arm_pick_object(obj, self.robot, action)
            elif operator == 'two_arm_place':
                two_arm_place_object(obj, self.robot, action)
            elif operator == 'one_arm_pick':
                one_arm_pick_object(obj, self.robot, action)
            elif operator == 'one_arm_place':
                one_arm_place_object(obj, self.robot, action)


    def visualize_plan(self, plan):
        print "Visualizing plan"
        for plan_step in plan:
            obj = self.env.GetKinBody(plan_step['obj_name'])
            action = plan_step['action']
            operator = plan_step['operator']
            subpath = plan_step['path']

            is_packing_action = isinstance(subpath, dict) and 'path_back_to_fetched_base_pose' in subpath.keys()
            is_subpath_crp = isinstance(subpath, dict) and not is_packing_action

            if is_subpath_crp:
                self.visualize_crp(subpath)
            elif is_packing_action:
                place_base_pose = action
                visualize_path(self.robot, subpath['place_path'])
                self.place_object(place_base_pose, obj)
                visualize_path(self.robot, subpath['path_back_to_fetched_base_pose'])
                set_robot_config(subpath['path_back_to_fetched_base_pose'][-1], self.robot)
            else:
                if operator == 'two_arm_pick':
                    full_body_pick_config = subpath[0]
                    pick_path = subpath[1]
                    self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
                    visualize_path(self.robot, pick_path)
                    pick_base_pose = action[0]
                    set_robot_config(pick_base_pose, self.robot)
                    self.robot.SetDOFValues(full_body_pick_config)
                    grab_obj(self.robot, obj)
                elif operator == 'two_arm_place':
                    place_base_pose = action
                    place_path = subpath
                    visualize_path(self.robot, place_path)
                    self.place_object(place_base_pose, obj)

    def get_region_containing(self, obj):
        containing_regions = []
        for r in self.regions.values():
            if r.name == 'entire_region':
                continue
            if r.contains(obj.ComputeAABB()):
                containing_regions.append(r)

        if len(containing_regions) == 0:
            return None
        elif len(containing_regions) == 1:
            return containing_regions[0]
        else:
            region_with_smallest_area = containing_regions[0]
            for r in containing_regions:
                if r.area() < region_with_smallest_area.area():
                    region_with_smallest_area = r
            return region_with_smallest_area

    def apply_pick_constraint(self, obj_name, pick_config, pick_base_pose=None):
        # todo I think this function can be removed?
        obj = self.env.GetKinBody(obj_name)
        if pick_base_pose is not None:
            set_robot_config(pick_base_pose, self.robot)
        self.robot.SetDOFValues(pick_config)
        grab_obj(self.robot, obj)

    def disable_objects_in_region(self, region_name):
        raise NotImplementedError

    def enable_objects_in_region(self, region_name):
        raise NotImplementedError

    def is_region_contains_all_objects(self, region, objects):
        return np.all([region.contains(obj.ComputeAABB()) for obj in objects])


    def get_objs_in_collision(self, path, region_name):
        assert len(path[0]) == self.robot.GetActiveDOF(), 'Robot active dof should match the path'
        objs = self.get_objs_in_region(region_name)
        in_collision = []
        with self.robot:
            for conf in path:
                set_active_dof_conf(conf, self.robot)
                #set_robot_config(conf, self.robot)
                if self.env.CheckCollision(self.robot):
                    for obj in objs:
                        if self.env.CheckCollision(self.robot, obj) and obj not in in_collision:
                            in_collision.append(obj)
        return in_collision


    """
    def compute_g_config(self, pick_base_pose, grasp_params, obj_to_pick):
        set_robot_config(pick_base_pose, self.robot)
        grasps = compute_two_arm_grasp(depth_portion=grasp_params[2],
                                       height_portion=grasp_params[1],
                                       theta=grasp_params[0],
                                       obj=obj_to_pick,
                                       robot=self.robot)
        g_config = solveTwoArmIKs(self.env, self.robot, obj_to_pick, grasps)
    """

    def apply_pick_action(self, action, obj=None):
        raise NotImplementedError

    def update_next_obj_to_pick(self, place_action):
        raise NotImplementedError

    def apply_place_action(self, action, do_check_reachability=True):
        raise NotImplementedError

    def remove_all_obstacles(self):
        raise NotImplementedError

    def is_goal_reached(self):
        raise NotImplementedError

    def set_init_state(self, saver):
        raise NotImplementedError

    def replay_plan(self, plan):
        raise NotImplementedError

    def which_operator(self, obj):
        raise NotImplementedError

    def restore(self, state_saver):
        raise NotImplementedError


class TwoArmProblemEnvironment(ProblemEnvironment):
    def __init__(self):
        ProblemEnvironment.__init__(self)

    def restore(self, state_saver):
        curr_obj = state_saver.curr_obj
        which_operator = state_saver.which_operator
        if not which_operator == 'two_arm_pick':
            grab_obj(self.robot, curr_obj)
        else:
            if len(self.robot.GetGrabbed()) > 0:
                release_obj(self.robot, self.robot.GetGrabbed()[0])
        state_saver.Restore()
        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    def which_operator(self, obj=None):
        if self.is_pick_time():
            return 'two_arm_pick'
        else:
            return 'two_arm_place'

