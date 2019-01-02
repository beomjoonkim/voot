import copy
import numpy as np

from problem_environment import ProblemEnvironment
from mover_problem import mover_problem
from openravepy import DOFAffine
from problem_env_feasibility_checker import FetchModeFeasibilityChecker, RAMOModeFeasibilityChecker

from utils import draw_robot_at_conf, remove_drawn_configs, set_robot_config, grab_obj, get_body_xytheta, \
    visualize_path, check_collision_except, one_arm_pick_object, set_active_dof_conf, release_obj, one_arm_place_object, \
    two_arm_pick_object, two_arm_place_object, one_arm_pick_object
from manipulation.regions import create_region, AARegion

from planners.mcts_utils import make_action_hashable, is_action_hashable


class Mover(ProblemEnvironment):
    def __init__(self):
        ProblemEnvironment.__init__(self)
        self.problem_config = mover_problem(self.env)
        self.infeasible_reward = -2

        self.regions = {}
        self.regions['home_region'] = self.problem_config['home_region']
        self.regions['loading_region'] = self.problem_config['loading_region']
        self.regions['entire_region'] = self.problem_config['entire_region']
        self.regions['bridge_region'] = self.problem_config['bridge_region']
        for shelf_region in self.problem_config['shelf_regions'].values():
            self.regions[shelf_region.name] = shelf_region

        self.shelf_regions = self.problem_config['shelf_regions']
        self.box_regions = self.problem_config['box_regions']
        self.shelf_objs = self.problem_config['shelf_objects'].values()[0]
        self.shelf_objs = []
        for temp_shelf_objs in self.problem_config['shelf_objects'].values():
            self.shelf_objs += temp_shelf_objs

        self.small_objs = self.problem_config['objects_to_pack']
        self.big_objs = self.problem_config['big_objects_to_pack']
        self.packing_boxes = self.problem_config['packing_boxes']

        # related to fetching sub-problem
        self.is_solving_fetching = False

        # related to constraint-removal sub-problem
        self.is_solving_namo = False
        self.high_level_planner = None

        # related to packing sub-problem
        self.is_solving_packing = False
        self.fetch_base_config = None

        self.robot = self.env.GetRobots()[0]
        self.is_boxes_in_home = False
        self.is_big_objs_in_truck = False
        self.is_small_objs_in_boxes = False
        self.is_shelf_objs_in_boxes = False

        self.fetch_planner = None
        self.namo_planner = None

        self.name == 'mover'

    def disable_objects_in_region(self, region_name):
        # todo do it for the shelf objects too
        movable_objs = self.small_objs + self.big_objs + self.packing_boxes + self.shelf_objs
        for obj in movable_objs:
            if self.regions[region_name].contains(obj.ComputeAABB()):
                obj.Enable(False)

    def enable_objects_in_region(self, region_name):
        movable_objs = self.small_objs + self.big_objs + self.packing_boxes + self.shelf_objs
        for obj in movable_objs:
            if self.regions[region_name].contains(obj.ComputeAABB()):
                obj.Enable(True)

    def update_box_region(self, box):
        box_region = AARegion.create_on_body(box)
        box_region.color = (1., 1., 0., 0.25)
        self.box_regions[box.GetName()] = box_region

    def reset_to_init_state(self, node):
        saver = node.state_saver

        movable_objs = self.small_objs + self.big_objs + self.packing_boxes + self.shelf_objs
        obj_enable_status = [obj.IsEnabled() for obj in movable_objs]
        saver.Restore() # this call re-enables objects that are disabled
        for status, obj in zip(obj_enable_status, movable_objs):
            obj.Enable(status)

        obj = node.obj
        if node.operator.find('pick') == -1 and not self.is_solving_packing:
            grab_obj(self.robot, obj)

        if self.is_solving_namo:
            self.namo_planner.reset()

        if self.is_solving_packing:
            for obj in self.objs_to_pack:
                obj.Enable(False)
        self.high_level_planner.reset_task_plan_indices()
        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    def which_operator(self, obj):
        # todo put assert statement on the obj type
        if obj in self.big_objs + self.packing_boxes:
            if self.is_pick_time():
                return 'two_arm_pick'
            else:
                return 'two_arm_place'
        else:
            if self.is_pick_time():
                return 'one_arm_pick'
            else:
                return 'one_arm_place'

    def determine_reward(self, operator_name, obj, motion_plan, motion_plan_status, new_namo_obj_names=None):
        objs_in_collision = []
        if motion_plan_status == 'HasSolution':
            if self.is_solving_fetching:
                fetching_region = self.get_region_containing(self.robot) # todo: perhaps I should use entire_region
                if operator_name.find('two_arm') != -1:
                    objs_in_collision = self.get_objs_in_collision(motion_plan, fetching_region.name)
                    reward = np.exp(-len(objs_in_collision))
                else:
                    if operator_name == 'one_arm_pick':
                        assert len(self.robot.GetGrabbed()) == 0, 'Robot must not be grasping when calculating reward for picking'
                        objs_in_collision = self.get_objs_in_collision(motion_plan, fetching_region.name)
                        reward = np.exp(-len(objs_in_collision))
                        """
                        base_motion = motion_plan['base_motion']
                        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
                        base_motion_collision = self.get_objs_in_collision(base_motion, fetching_region.name)
                        reward = np.exp(-len(base_motion_collision))

                        arm_motion = motion_plan['arm_motion']
                        manip = self.robot.GetManipulator('rightarm_torso')
                        self.robot.SetActiveDOFs(manip.GetArmIndices())

                        with self.robot:
                            set_robot_config(base_motion[-1], self.robot)
                            self.robot.SetActiveDOFs(manip.GetArmIndices())
                            arm_motion_collision = self.get_objs_in_collision(arm_motion, fetching_region.name)
                            reward += np.exp(-len(arm_motion_collision))
                        objs_in_collision = base_motion_collision + arm_motion_collision
                        """
                    else:
                        objs_in_collision = self.get_objs_in_collision(motion_plan, fetching_region.name)
                        reward = np.exp(-len(objs_in_collision))
            elif self.is_solving_namo:
                if operator_name == 'two_arm_place':
                    reward = len(self.namo_planner.prev_namo_object_names) - len(new_namo_obj_names)
                    try:
                        objs_in_collision = [self.env.GetKinBody(name) for name in new_namo_obj_names]
                    except:
                        import pdb;pdb.set_trace()
                else:
                    objs_in_collision = [self.env.GetKinBody(name) for name in self.namo_planner.curr_namo_object_names]
                    reward = 0.5
            elif self.is_solving_packing:
                if operator_name == 'two_arm_place':
                    reward = 1
                else:
                    reward = 0.5
            else:
                assert False, "One of the sub-problem modes have to be on"
        else:
            reward = self.infeasible_reward

        return reward, objs_in_collision

    ### Operator instance application functions
    def apply_two_arm_pick_action(self, action, obj, region, check_feasibility, parent_motion):
        if action['g_config'] is None:
            curr_state = self.get_state()
            return curr_state, self.infeasible_reward, None
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
                new_namo_obj = node.children[make_action_hashable(action)].objs_in_collision
                new_namo_obj_names = [namo_obj.GetName() for namo_obj in new_namo_obj]
                self.namo_planner.prev_namo_object_names = [namo_obj.GetName() for namo_obj in node.parent.objs_in_collision]
                self.namo_planner.curr_namo_object_names = [namo_obj.GetName() for namo_obj in new_namo_obj]

                self.namo_planner.fetch_pick_path = node.children[make_action_hashable(action)].parent_motion['fetching_pick_motion']
                self.namo_planner.fetch_place_path = node.children[make_action_hashable(action)].parent_motion['fetching_place_motion']
                # todo what about the current fetch_pick_path and fetch_place_path when I reset, and get here?
                # -> they are stored on plan
            # what to do with self.namo_planner.prev_namo_object_names? New namo obj names?

        reward, objs_in_collision = self.determine_reward('two_arm_place', target_obj, plan, status, new_namo_obj_names)
        if status == 'HasSolution':
            two_arm_place_object(target_obj, self.robot, action)
            if target_obj.GetName().find('packing_box') != -1:
                self.update_box_region(target_obj)
        else:
            curr_state = self.get_state()

        return curr_state, reward, plan, objs_in_collision

    def apply_one_arm_pick_action(self, action, obj, region, check_feasibility, parent_motion):
        curr_state = self.get_state()
        pick_base_pose = action['base_pose']
        g_config = action['g_config']
        if g_config is None:
            return curr_state, self.infeasible_reward, None, None

        obj_to_pick = obj
        if check_feasibility:
            if self.is_solving_fetching:
                motion, status = self.fetch_planner.check_one_arm_pick_feasibility(obj, action)
            elif self.is_solving_namo:
                motion, status = self.namo_planner.check_one_arm_pick_feasibility(obj, action)
        else:
            motion = parent_motion
            status = 'HasSolution'
        reward, objs_in_collision = self.determine_reward('one_arm_pick', obj, motion, status)
        if status == 'HasSolution':
            one_arm_pick_object(obj, self.robot, action)
            curr_state = self.get_state()
        else:
            curr_state = self.get_state()

        return curr_state, reward, motion, objs_in_collision

    def apply_one_arm_place_action(self, action, obj, obj_placement_region, check_feasibility, parent_motion):
        # todo:
        #   Should I have a hall-way region? Let's consider a case where there is none.
        curr_state = self.get_state()
        if check_feasibility:
            if self.is_solving_fetching:
                motion, status = self.fetch_planner.check_one_arm_place_feasibility(obj, action, obj_placement_region)
            elif self.is_solving_namo:
                motion, status, new_namo_obj_names = self.namo_planner.check_one_arm_place_feasibility(obj, action, obj_placement_region)
        else:
            motion = parent_motion
            status = 'HasSolution'

        reward, objs_in_collision = self.determine_reward('one_arm_place', obj, motion, status)
        if status == 'HasSolution':
            # todo move this to one_arm_place function
            """
            manip = self.robot.GetManipulator('rightarm_torso')
            self.robot.SetActiveDOFs(manip.GetArmIndices(), DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
            base_pose = action['base_pose']
            g_config = action['g_config']
            full_place_config = np.hstack([g_config, base_pose.squeeze()])
            set_active_dof_conf(full_place_config, self.robot)
            """
            one_arm_place_object(obj, self.robot, action)
            curr_state = self.get_state()

        return curr_state, reward, motion, objs_in_collision

    # Related to checking feasibility of operator instances
    def check_base_pose_feasible(self, base_pose, obj, region):
        if region.name == 'bridge_region':
            if not self.is_collision_at_base_pose(base_pose, obj) \
                    and self.is_in_region_at_base_pose(base_pose, obj, robot_region=self.regions['entire_region'],
                                                   obj_region=region):
                return True
        else:
            if not self.is_collision_at_base_pose(base_pose, obj) \
                    and self.is_in_region_at_base_pose(base_pose, obj, robot_region=region,
                                                       obj_region=region):
                return True
        return False

    ## helper functons

    def get_objs_in_region(self, region_name):
        movable_objs = self.small_objs + self.big_objs + self.packing_boxes + self.shelf_objs
        objs_in_region = []
        for obj in movable_objs:
            if self.regions[region_name].contains(obj.ComputeAABB()):
                objs_in_region.append(obj)
        return objs_in_region


    def set_arm_base_config(self, config):
        manip = self.robot.GetManipulator('rightarm_torso')
        self.robot.SetActiveDOFs(manip.GetArmIndices(), DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
        set_active_dof_conf(config, self.robot)

    ########################################################################################


