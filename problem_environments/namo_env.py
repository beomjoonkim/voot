import numpy as np
import sys
import socket
import os
import random
import copy

from openravepy import DOFAffine
from planners.mcts_utils import make_action_hashable, is_action_hashable
from manipulation.primitives.savers import DynamicEnvironmentStateSaver
## NAMO problem environment
from problem_environment import ProblemEnvironment
from NAMO_problem import NAMO_problem

## openrave_wrapper imports
from manipulation.bodies.bodies import set_color

## mover library utility functions
sys.path.append('../mover_library/')
from utils import set_robot_config, get_body_xytheta, check_collision_except,  grab_obj, \
    simulate_path, two_arm_pick_object, two_arm_place_object
from operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp

OBJECT_ORIGINAL_COLOR = (0, 0, 0)
COLLIDING_OBJ_COLOR = (0, 1, 1)
TARGET_OBJ_COLOR = (1, 0, 0)


class NAMO(ProblemEnvironment):
    def __init__(self):
        ProblemEnvironment.__init__(self)

        self.problem_config = NAMO_problem(self.env)
        self.robot = self.env.GetRobots()[0]

        self.objects = self.problem_config['objects']
        for obj in self.objects:
            set_color(obj, OBJECT_ORIGINAL_COLOR)

        self.target_object = self.problem_config['target_obj']
        set_color(self.target_object, TARGET_OBJ_COLOR)

        self.regions = {'entire_region': self.problem_config['entire_region'],
                        'loading_region': self.problem_config['loading_region']}

        self.init_base_conf = np.array([-1, 1, 0])
        self.infeasible_reward = -2
        self.is_init_pick_node = True
        self.name = 'namo'
        self.init_saver = self.problem_config['initial_saver']

    def get_objs_in_region(self, region_name):
        movable_objs = self.objects
        objs_in_region = []
        for obj in movable_objs:
            if self.regions[region_name].contains(obj.ComputeAABB()):
                objs_in_region.append(obj)
        return objs_in_region

    def get_region_containing(self, obj):
        return self.regions['entire_region']

    def determine_reward(self, operator_name, obj, motion_plan, motion_plan_status, new_namo_obj_names=None):
        objs_in_collision = []

        if motion_plan_status == 'HasSolution':
            if self.is_solving_fetching:
                fetching_region = self.get_region_containing(self.robot)
                if operator_name.find('two_arm') != -1:
                    objs_in_collision = self.get_objs_in_collision(motion_plan, fetching_region.name)
                    reward = np.exp(-len(objs_in_collision))
            elif self.is_solving_namo:
                if operator_name == 'two_arm_place':
                    reward = len(self.namo_planner.prev_namo_object_names) - len(new_namo_obj_names)
                    objs_in_collision = [self.env.GetKinBody(name) for name in new_namo_obj_names]
                else:
                    objs_in_collision = [self.env.GetKinBody(name) for name in self.namo_planner.curr_namo_object_names]
                    reward = 0.5
        else:
            reward = self.infeasible_reward

        return reward, objs_in_collision

    def reset_to_init_state(self, node):
        saver = node.state_saver
        saver.Restore()  # this call re-enables objects that are disabled
        self.curr_state = self.get_state()
        self.placements = copy.deepcopy(self.initial_placements)

        if not self.init_which_opreator != 'two_arm_pick':
            grab_obj(self.robot, self.curr_obj)
        if self.is_solving_namo:
            self.namo_planner.reset()
        self.high_level_planner.reset_task_plan_indices()
        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    def disable_objects_in_region(self, region_name):
        for object in self.objects:
            object.Enable(False)

    def enable_objects_in_region(self, region_name):
        for object in self.objects:
            object.Enable(True)

    def check_base_pose_feasible(self, base_pose, obj, region):
        if not self.is_collision_at_base_pose(base_pose, obj) \
                and self.is_in_region_at_base_pose(base_pose, obj, robot_region=region,
                                                   obj_region=region):
                return True
        return False

    def apply_two_arm_pick_action(self, action, obj, region, check_feasibility, parent_motion):
        motion_plan = None
        status = "NoSolution"
        if action['g_config'] is None:
            curr_state = self.get_state()
            namo_objects = [self.env.GetKinBody(namo_obj_name) for namo_obj_name in self.namo_planner.curr_namo_object_names]
            return curr_state, self.infeasible_reward, None, namo_objects

        if check_feasibility:
            if self.is_solving_fetching:
                motion_plan, status = self.fetch_planner.check_two_arm_pick_feasibility(obj, action, region)
            elif self.is_solving_namo:
                motion_plan, status = self.namo_planner.check_two_arm_pick_feasibility(obj, action)
        else:
            motion_plan = parent_motion
            status = 'HasSolution'

        reward, objs_in_collision = self.determine_reward('two_arm_pick', obj, motion_plan, status)
        if status == 'HasSolution':
            two_arm_pick_object(obj, self.robot, action)
            curr_state = self.get_state()
            if self.is_solving_namo:
                # update the current pick configuration
                self.namo_planner.current_pick_conf = action['base_pose']
        else:
            curr_state = self.get_state()

        return curr_state, reward, motion_plan, objs_in_collision

    def apply_two_arm_place_action(self, action, node, check_feasibility, parent_motion):
        target_obj = node.obj
        target_region = node.region

        base_pose = action['base_pose']
        curr_state = self.get_state()
        new_namo_obj_names = None
        if check_feasibility:
            if self.is_solving_fetching:
                plan, status = self.fetch_planner.check_two_arm_place_feasibility(target_obj, action, target_region)
            elif self.is_solving_namo:
                plan, status, new_namo_obj_names = self.namo_planner.check_two_arm_place_feasibility(target_obj, action, target_region)
        else:
            status = 'HasSolution'
            plan = parent_motion
            if self.is_solving_namo:
                new_namo_objs = node.children[make_action_hashable(action)].objs_in_collision
                new_namo_obj_names = [namo_obj.GetName() for namo_obj in new_namo_objs]
                self.namo_planner.prev_namo_object_names = [namo_obj.GetName() for namo_obj in node.parent.objs_in_collision]
                self.namo_planner.curr_namo_object_names = [namo_obj.GetName() for namo_obj in new_namo_objs]

                self.namo_planner.fetch_pick_path = node.children[make_action_hashable(action)].parent_motion['fetch_pick_path']
                self.namo_planner.fetch_place_path = node.children[make_action_hashable(action)].parent_motion['fetch_place_path']

                # todo update the task-plan?
                self.high_level_planner.set_task_plan([{'region': self.regions['entire_region'], 'objects': new_namo_objs}])

        reward, objs_in_collision = self.determine_reward('two_arm_place', target_obj, plan, status, new_namo_obj_names)
        if status == 'HasSolution':
            two_arm_place_object(target_obj, self.robot, action)
            curr_state = self.get_state()
            if self.is_solving_namo:
                set_robot_config(self.namo_planner.current_pick_conf, self.robot)
        else:
            curr_state = self.get_state()

        return curr_state, reward, plan, objs_in_collision

    def is_goal_reached(self):
        return len(self.objs_to_move) == 0

    def which_operator(self, obj=None):
        if self.is_pick_time():
            return 'two_arm_pick'
        else:
            return 'two_arm_place'

    def apply_two_arm_pick_action_stripstream(self, action, obj=None, do_check_reachability=False):
        if obj is None:
            obj_to_pick = self.curr_obj
        else:
            obj_to_pick = obj

        pick_base_pose, grasp_params, g_config = action
        set_robot_config(pick_base_pose, self.robot)
        """
        grasps = compute_two_arm_grasp(depth_portion=grasp_params[2],
                                       height_portion=grasp_params[1],
                                       theta=grasp_params[0],
                                       obj=obj_to_pick,
                                       robot=self.robot)
        g_config = solveTwoArmIKs(self.env, self.robot, obj_to_pick, grasps)
        try:
            assert g_config is not None
        except:
            import pdb;pdb.set_trace()
        """

        action = {'base_pose': pick_base_pose, 'g_config': g_config}
        two_arm_pick_object(obj_to_pick, self.robot, action)
        curr_state = self.get_state()
        reward = 0
        pick_path = None
        return curr_state, reward, g_config, pick_path







