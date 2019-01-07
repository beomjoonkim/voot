import sys

from openravepy import DOFAffine
from manipulation.primitives.savers import DynamicEnvironmentStateSaver
from namo_planner import NAMOPlanner
sys.path.append('../mover_library/')
from utils import draw_robot_at_conf, remove_drawn_configs, set_robot_config, grab_obj, get_body_xytheta, \
    visualize_path, check_collision_except, one_arm_pick_object, set_active_dof_conf, release_obj, one_arm_place_object, \
    two_arm_pick_object, two_arm_place_object, set_obj_xytheta, two_arm_place_object


import copy
import numpy as np


class NamoDomainNamoPlanner(NAMOPlanner):
    def __init__(self, problem_env, high_level_controller, n_iter, n_optimal_iter, max_time):
        NAMOPlanner.__init__(self, problem_env, high_level_controller, n_iter, n_optimal_iter)
        self.init_namo_object_names_on_place_path = []
        self.current_pick_conf = None
        self.max_time = max_time

    def get_obstacles_in_collision_from_fetching_path(self, pick_path, place_path):
        curr_region = self.problem_env.get_region_containing(self.problem_env.robot)
        assert len(self.robot.GetGrabbed()) == 0, 'this function assumes the robot starts with object in hand'

        ### fetch place path collisions
        if self.fetch_place_op_instance is not None:
            two_arm_pick_object(self.fetching_obj, self.robot, self.fetch_pick_op_instance['action'])
            place_collisions = self.problem_env.get_objs_in_collision(place_path, curr_region.name)

        ### fetch pick path collisions
        two_arm_place_object(self.fetching_obj, self.robot, self.fetch_pick_op_instance['action'])
        pick_collisions = self.problem_env.get_objs_in_collision(pick_path, curr_region.name)

        collisions = pick_collisions
        if self.fetch_place_op_instance is not None:
            collisions += [p for p in place_collisions if p not in collisions]

        # go back to the original robot state
        two_arm_place_object(self.fetching_obj, self.problem_env.robot, self.fetch_pick_op_instance['action'])

        return collisions

    def get_motion_plan_with_disabling(self, goal, motion_planning_region_name, is_one_arm, exception_obj=None):
        curr_region = self.problem_env.get_region_containing(self.problem_env.robot)
        motion, status = self.problem_env.get_base_motion_plan(goal, motion_planning_region_name, n_iterations=[20])

        if status == "NoPath":
            self.problem_env.disable_objects_in_region(curr_region.name)
            self.fetching_obj.Enable(True)
            if exception_obj is not None:
                exception_obj.Enable(True)
            motion, status = self.problem_env.get_base_motion_plan(goal, motion_planning_region_name)
            self.problem_env.enable_objects_in_region(curr_region.name)

        return motion, status

    def get_new_fetching_pick_path(self, namo_obj, motion_planning_region_name):
        # todo plan to the previous namo_obj pick configuration
        motion_to_namo_pick, status1 = self.problem_env.get_base_motion_plan(self.current_pick_conf,
                                                                             motion_planning_region_name)
        if status1 == 'NoPath':
            return None, 'NoPath'

        set_robot_config(self.current_pick_conf, self.robot)
        motion_from_namo_pick_to_fetch_pick, status2 = self.get_motion_plan_with_disabling(self.fetch_pick_conf,
                                                                                           motion_planning_region_name,
                                                                                           False, namo_obj)
        if status2 == 'NoPath':
            return None, 'NoPath'

        motion = motion_to_namo_pick + motion_from_namo_pick_to_fetch_pick
        status = "HasSolution"
        return motion, status

    def check_two_arm_pick_feasibility(self, obj, action):
        motion_planning_region = self.problem_env.get_region_containing(obj)
        goal_robot_xytheta = action['base_pose']

        motion = None
        status = "NoPath"
        if action['g_config'] is not None and self.problem_env.check_base_pose_feasible(goal_robot_xytheta, obj,
                                                                                        motion_planning_region):
            motion, status = self.problem_env.get_base_motion_plan(goal_robot_xytheta, motion_planning_region.name)

        if action['g_config'] is None or status == 'NoPath':
            motion = None
            status = 'NoPath'
            #self.get_new_fetching_pick_path(obj, motion_planning_region.name)

        return motion, status


    def check_two_arm_place_feasibility(self, namo_obj, action, obj_placement_region):
        motion_planning_region_name = 'entire_region'

        goal_robot_xytheta = action['base_pose']
        pick_base_pose = get_body_xytheta(self.problem_env.robot)
        pick_conf = self.problem_env.robot.GetDOFValues()


        namo_status = 'NoPath'
        namo_place_motion = None
        if self.problem_env.check_base_pose_feasible(goal_robot_xytheta, namo_obj,
                                                     self.problem_env.regions[motion_planning_region_name]):
            namo_place_motion, namo_status = self.problem_env.get_base_motion_plan(goal_robot_xytheta,
                                                                                   motion_planning_region_name)
        if namo_status == 'NoPath':
            return namo_place_motion, namo_status, self.curr_namo_object_names
        two_arm_place_object(namo_obj, self.problem_env.robot, action)

        fetch_pick_path = self.fetch_pick_path
        fetch_place_path = self.fetch_place_path
        new_collisions = self.get_obstacles_in_collision_from_fetching_path(fetch_pick_path, fetch_place_path)

        # go back to pre-place
        self.problem_env.robot.SetDOFValues(pick_conf)
        set_robot_config(action['base_pose'], self.robot)
        grab_obj(self.problem_env.robot, namo_obj)
        set_robot_config(pick_base_pose, self.robot)

        """
        if namo_obj in new_collisions:
            print "Object moved still in collision"
            return None, "NoPath", self.curr_namo_object_names
        """

        # if new collisions is more than or equal to the current collisions, don't bother executing it
        if len(self.curr_namo_object_names) <= len(new_collisions):
            print "There are more or equal number of collisions on the new path"
            print len(self.curr_namo_object_names), len(new_collisions)
            print namo_obj, new_collisions
            return None, "NoPath", self.curr_namo_object_names

        # otherwise, update the new namo objects
        self.prev_namo_object_names = self.curr_namo_object_names
        self.curr_namo_object_names = [obj.GetName() for obj in new_collisions]
        if len(self.prev_namo_object_names) - len(self.curr_namo_object_names ) > 1:
            import pdb;pdb.set_trace()


        # pick motion is the path to the fetching object, place motion is the path to place the namo object
        motion = {'fetch_pick_path': fetch_pick_path, 'fetch_place_path': fetch_place_path,
                  'place_motion': namo_place_motion}

        # update the self.fetch_place_path if the packing region has moved
        self.fetch_pick_path = fetch_pick_path
        self.fetch_place_path = fetch_place_path

        # update the high level controller task plan
        namo_region = self.problem_env.get_region_containing(self.fetching_obj)
        self.high_level_controller.set_task_plan([{'region': namo_region, 'objects': new_collisions}])
        return motion, "HasSolution", self.curr_namo_object_names

    def namo_domain_solve_single_object(self, initial_collision_names, mcts):
        # get the initial collisions
        pick_collisions = initial_collision_names['pick_collisions']
        place_collisions = initial_collision_names['place_collisions']
        initial_collision_names = pick_collisions
        initial_collision_names += [tmp for tmp in place_collisions if tmp not in pick_collisions]
        self.init_namo_object_names_on_place_path = place_collisions
        self.init_namo_object_names = initial_collision_names
        self.curr_namo_object_names = copy.deepcopy(self.init_namo_object_names)

        curr_namo_region = 'entire_region'
        namo_problem_for_obj = [{'region': curr_namo_region, 'objects': [self.env.GetKinBody(obj_name)
                                            for obj_name in self.curr_namo_object_names]}]
        self.high_level_controller.set_task_plan(namo_problem_for_obj)
        mcts.update_init_node_obj()
        self.problem_env.high_level_planner = self.high_level_controller

        # setup the task plan
        prenamo_initial_node = self.fetch_goal_node
        namo_plan = None
        goal_node = None
        self.problem_env.namo_planner = self
        self.high_level_controller.set_task_plan(namo_problem_for_obj)
        mcts.update_init_node_obj()

        mcts.switch_init_node(prenamo_initial_node)
        self.problem_env.is_solving_namo = True
        search_time_to_reward, namo_plan, goal_node = mcts.search(n_iter=self.n_iter,
                                                                  n_optimal_iter=self.n_optimal_iter,
                                                                  max_time=self.max_time)
        self.problem_env.is_solving_namo = False

        return search_time_to_reward, namo_plan, goal_node

    def namo_domain_initialize_namo_problem(self, fetch_plan, fetch_goal_node):
        self.fetching_obj = self.env.GetKinBody(fetch_plan[0]['obj_name'])

        self.fetch_pick_op_instance = fetch_plan[0]
        #self.fetch_place_op_instance = fetch_plan[1]
        self.fetch_pick_path = fetch_plan[0]['path']
        #self.fetch_place_path = fetch_plan[1]['path']

        self.fetch_pick_conf = self.problem_env.make_config_from_op_instance(self.fetch_pick_op_instance)
        #self.fetch_place_conf = self.problem_env.make_config_from_op_instance(self.fetch_place_op_instance)
        self.problem_env.high_level_planner = self.high_level_controller
        self.fetch_goal_node = fetch_goal_node
        self.fetch_goal_node.state_saver = DynamicEnvironmentStateSaver(self.env) # this assumes that we're in prefetch state
        self.prefetching_robot_config = get_body_xytheta(self.problem_env.robot).squeeze()

    ########################################################################################

