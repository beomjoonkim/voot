import sys
import numpy as np

sys.path.append('../mover_library/')
from utils import visualize_path, two_arm_pick_object, one_arm_pick_object


class FetchPlanner:
    def __init__(self, problem_env, high_level_controller):
        self.problem_env = problem_env
        self.high_level_controller = high_level_controller
        self.fetching_object = None

        self.env = self.problem_env.env
        self.robot = self.problem_env.robot

    def get_motion_plan_with_disabling(self, goal, motion_planning_region_name, is_one_arm, exception_obj=None):
        curr_region = self.problem_env.get_region_containing(self.problem_env.robot)
        # todo what about when moving objects inside the shelf?

        # todo check feasible base
        if is_one_arm:
            self.fetching_object.Enable(False)
            motion, status = self.problem_env.get_arm_base_motion_plan(goal, motion_planning_region_name)
            self.fetching_object.Enable(True)
        else:
            motion, status = self.problem_env.get_base_motion_plan(goal, motion_planning_region_name, n_iterations=[20])

        if status == "NoPath":
            self.problem_env.disable_objects_in_region(curr_region.name)
            self.fetching_object.Enable(True)
            if exception_obj is not None:
                exception_obj.Enable(True)
            if is_one_arm:
                motion, status = self.problem_env.get_arm_base_motion_plan(goal, motion_planning_region_name)
            else:
                motion, status = self.problem_env.get_base_motion_plan(goal, motion_planning_region_name)
            self.problem_env.enable_objects_in_region(curr_region.name)
            if status == "NoPath":
                return None, "NoPath"
        return motion, status

    def solve_fetching_single_object(self, target_object, target_region, mcts, init_node):
        connecting_region = self.get_connecting_region(target_region)
        self.fetching_object = target_object[0]
        self.problem_env.is_solving_fetching = True
        self.problem_env.fetch_planner = self
        self.problem_env.high_level_planner = self.high_level_controller
        self.high_level_controller.task_plan = [{'region': connecting_region, 'objects': target_object}]
        if init_node is not None:
            mcts.switch_init_node(init_node)
        search_time_to_reward, fetch_plan, goal_node = mcts.search(self.high_level_controller.n_iter,
                                                                   self.high_level_controller.n_optimal_iter,
                                                                   self.high_level_controller.max_time)
        self.problem_env.is_solving_fetching = False
        self.high_level_controller.reset_task_plan_indices()
        return search_time_to_reward, fetch_plan, goal_node

    def solve_packing(self, target_objects, target_region, mcts, init_node):
        connecting_region = self.get_connecting_region(target_region)
        #self.fetching_object = target_object
        self.problem_env.is_solving_fetching = True
        self.problem_env.fetch_planner = self
        self.problem_env.high_level_planner = self.high_level_controller
        self.high_level_controller.task_plan = [{'region': connecting_region, 'objects': target_objects}]
        if init_node is not None:
            mcts.switch_init_node(init_node)
        search_time_to_reward, fetch_plan, goal_node = mcts.search(self.high_level_controller.n_iter,
                                                                   self.high_level_controller.n_optimal_iter,
                                                                   self.high_level_controller.max_time)
        self.problem_env.is_solving_fetching = False
        self.high_level_controller.reset_task_plan_indices()
        return search_time_to_reward, fetch_plan, goal_node

    def check_one_arm_pick_feasibility(self, obj, action):
        base_pose = action['base_pose']
        g_config = action['g_config']
        if g_config is None:
            return None, "NoPath"

        full_pick_config = np.hstack([g_config, base_pose.squeeze()])
        curr_obj_region = self.problem_env.get_region_containing(obj)
        curr_robot_region = self.problem_env.get_region_containing(self.problem_env.robot)
        if curr_obj_region.name.find('shelf') != -1:
            motion_planning_region = curr_robot_region
        else:
            motion_planning_region = curr_obj_region

        is_base_feasible = self.problem_env.check_base_pose_feasible(base_pose, obj, curr_obj_region)

        motion = None
        status = "NoPath"
        if is_base_feasible:
            motion, status = self.problem_env.get_arm_base_motion_plan(full_pick_config, motion_planning_region.name)

        if not is_base_feasible or status != 'HasSolution':
            self.problem_env.disable_objects_in_region(motion_planning_region.name)
            obj.Enable(True)
            motion, status = self.problem_env.get_arm_base_motion_plan(full_pick_config, motion_planning_region.name)
            self.problem_env.enable_objects_in_region(motion_planning_region.name)

        return motion, status

    def check_one_arm_place_feasibility(self, obj, action, placement_region):
        base_pose = action['base_pose']
        g_config = action['g_config']
        if g_config is None:
            return None, 'NoPath'
        motion_planning_region = self.problem_env.get_region_containing(self.problem_env.robot)
        full_place_config = np.hstack([g_config, base_pose.squeeze()])
        obj_held = self.problem_env.robot.GetGrabbed()[0]

        is_base_feasible = self.problem_env.check_base_pose_feasible(base_pose, obj, motion_planning_region)

        motion = None
        status = "NoPath"
        if is_base_feasible:
            motion, status = self.problem_env.get_arm_base_motion_plan(full_place_config, motion_planning_region.name)
        import pdb;pdb.set_trace()

        if not is_base_feasible or status != 'HasSolution':
            self.problem_env.disable_objects_in_region(motion_planning_region.name)
            obj_held.Enable(True)
            #if placement_region.name.find('packing_box') != -1:
            #    packing_box = self.env.GetKinBody(placement_region.name)
            #    packing_box.Enable(True)
            # todo if the packing region is an object, then enable that too
            motion, status = self.problem_env.get_arm_base_motion_plan(full_place_config, motion_planning_region.name)
            self.problem_env.enable_objects_in_region(motion_planning_region.name)

        return motion, status

    def check_two_arm_pick_feasibility(self, obj, action, target_region):
        curr_region = self.problem_env.get_region_containing(obj)
        motion_planning_region_name = target_region.name if curr_region.name == target_region.name else 'entire_region'
        goal_robot_xytheta = action['base_pose']
        motion, status = self.get_motion_plan_with_disabling(goal_robot_xytheta, motion_planning_region_name, False)
        """
        #self.problem_env.disable_objects_in_region('bridge_region')
        if self.problem_env.check_base_pose_feasible(goal_robot_xytheta, obj, target_region):
            motion, status = self.problem_env.get_base_motion_plan(goal_robot_xytheta, motion_planning_region_name)
            if status == 'NoPath':
                self.problem_env.disable_objects_in_region(curr_region.name)
                obj.Enable(True)
                motion, status = self.problem_env.get_base_motion_plan(goal_robot_xytheta, motion_planning_region_name)
                self.problem_env.enable_objects_in_region(curr_region.name)
        else:
            self.problem_env.disable_objects_in_region(curr_region.name)
            obj.Enable(True)
            if self.problem_env.check_base_pose_feasible(goal_robot_xytheta, obj, target_region):
                motion, status = self.problem_env.get_base_motion_plan(goal_robot_xytheta, motion_planning_region_name)
            else:
                motion = None
                status = 'NoPath'
            # todo where does it enable the objects in the bridge region?
            self.problem_env.enable_objects_in_region(curr_region.name)
        """

        return motion, status

    def check_two_arm_place_feasibility(self, obj, action, target_region):
        # the code is exactly the same as pick
        return self.check_two_arm_pick_feasibility(obj, action, target_region)

    ################ Helper functions #########################
    def get_connecting_region(self, target_region):
        return target_region

    @staticmethod
    def get_initial_collisions(fetch_plan):
        pick_collisions = fetch_plan[0]['obj_names_in_collision']
        if len(fetch_plan) > 1:
            place_collisions = fetch_plan[1]['obj_names_in_collision']
        else:
            place_collisions = []
        collisions = {'pick_collisions': pick_collisions, 'place_collisions': place_collisions}
        return collisions

    @staticmethod
    def get_fetch_pick_config(fetch_plan):
        pick_base_config = fetch_plan[0]['action']['base_pose']
        pick_arm_conf = fetch_plan[0]['action']['g_config']
        if fetch_plan[0]['operator'] == 'one_arm_pick':
            pick_config = np.hstack([pick_arm_conf, pick_base_config.squeeze()])
        else:
            pick_config = pick_base_config
        return pick_config

    @staticmethod
    def get_fetch_place_config(fetch_plan):
        place_base_config = fetch_plan[1]['action']['base_pose']
        place_arm_conf = fetch_plan[1]['action']['g_config']
        if fetch_plan[1]['operator'] == 'one_arm_place':
            place_config = np.hstack([place_arm_conf, place_base_config.squeeze()])
        else:
            place_config = place_base_config
        return place_config

    def get_fetch_pick_entrance_config(self, fetch_plan, target_obj):
        fetch_pick = fetch_plan[0]
        fetch_motion = fetch_pick['path']
        fetching_region = self.problem_env.get_region_containing(target_obj)

        l_finger_tip = self.robot.GetLink('r_gripper_l_finger_tip_link')
        r_finger_tip = self.robot.GetLink('r_gripper_r_finger_tip_link')
        for pidx,p in enumerate(fetch_motion):
            self.problem_env.set_arm_base_config(p)
            if fetching_region.contains(l_finger_tip.ComputeAABB()) or fetching_region.contains(r_finger_tip.ComputeAABB()):
                break
        return fetch_motion[pidx-1]

    def get_fetch_pick_exit_config(self, fetch_plan, target_obj):
        fetch_place = fetch_plan[1]
        fetch_motion = fetch_place['path']
        fetching_region = self.problem_env.get_region_containing(target_obj)

        fetch_pick = fetch_plan[0]
        if fetch_pick['operator'] == 'one_arm_pick':
            one_arm_pick_object(target_obj, self.robot, fetch_pick['action'])
        else:
            two_arm_pick_object(target_obj, self.robot, fetch_pick['action'])

        for pidx, p in enumerate(fetch_motion):
            self.problem_env.set_arm_base_config(p)
            if not fetching_region.contains(target_obj.ComputeAABB()):
                break
        return fetch_motion[pidx + 1]

    def get_fetch_place_entrance_config(self, fetch_plan, target_obj, target_region):
        fetch_place = fetch_plan[1]
        fetch_motion = fetch_place['path']
        place_region = target_region

        fetch_pick = fetch_plan[0]
        if fetch_pick['operator'] == 'one_arm_pick':
            one_arm_pick_object(target_obj, self.robot, fetch_pick['action'])
        else:
            two_arm_pick_object(target_obj, self.robot, fetch_pick['action'])

        l_finger_tip = self.robot.GetLink('r_gripper_l_finger_tip_link')
        r_finger_tip = self.robot.GetLink('r_gripper_r_finger_tip_link')
        for pidx,p in enumerate(fetch_motion):
            self.problem_env.set_arm_base_config(p)
            if place_region.contains(target_obj.ComputeAABB()) \
                    or place_region.contains(l_finger_tip.ComputeAABB()) \
                    or place_region.contains(r_finger_tip.ComputeAABB()):
                break
        return fetch_motion[pidx - 1]


    @staticmethod
    def get_fetch_place_path(fetch_plan):
        return fetch_plan[1]['path']

    @staticmethod
    def get_fetch_pick_path(fetch_plan):
        return fetch_plan[0]['path']



