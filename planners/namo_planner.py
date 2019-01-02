import sys

from openravepy import DOFAffine
from manipulation.primitives.savers import DynamicEnvironmentStateSaver
sys.path.append('../mover_library/')
from utils import draw_robot_at_conf, remove_drawn_configs, set_robot_config, grab_obj, get_body_xytheta, \
    visualize_path, check_collision_except, one_arm_pick_object, set_active_dof_conf, release_obj, one_arm_place_object, \
    two_arm_pick_object, two_arm_place_object, set_obj_xytheta, two_arm_place_object

import copy
import numpy as np


class NAMOPlanner:
    def __init__(self, problem_env, high_level_controller, n_iter, n_optimal_iter):
        self.problem_env = problem_env
        self.high_level_controller = high_level_controller
        self.n_iter = n_iter
        self.n_optimal_iter = n_optimal_iter

        # related to solving the NAMO problem
        self.prefetching_robot_config = None
        self.curr_namo_object_names = None
        self.init_namo_object_names = None
        self.prev_namo_object_names = None
        self.fetch_pick_op_instance = None
        self.fetch_place_op_instance = None
        self.fetching_obj = None
        self.fetch_pick_conf = None
        self.fetch_place_conf = None
        self.fetch_place_path = None
        self.fetch_pick_path = None

        self.robot = self.problem_env.robot
        self.env = self.problem_env.env

    def get_motion_plan_with_disabling(self, goal, motion_planning_region_name, is_one_arm, exception_obj=None):
        curr_region = self.problem_env.get_region_containing(self.problem_env.robot)
        #todo what about when moving objects inside the shelf?

        # todo check feasible base
        if is_one_arm:
            self.fetching_obj.Enable(False)
            motion, status = self.problem_env.get_arm_base_motion_plan(goal, motion_planning_region_name)
            self.fetching_obj.Enable(True)
        else:
            motion, status = self.problem_env.get_base_motion_plan(goal, motion_planning_region_name)

        if status == "NoPath":
            self.problem_env.disable_objects_in_region(curr_region.name)
            self.fetching_obj.Enable(True)
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

    def reset(self):
        self.curr_namo_object_names = copy.deepcopy(self.init_namo_object_names)
        namo_objs = [self.problem_env.env.GetKinBody(name) for name in self.curr_namo_object_names]
        namo_region = self.problem_env.get_region_containing(self.fetching_obj)
        task_plan = [{'region': namo_region, 'objects': namo_objs}]
        self.high_level_controller.set_task_plan(task_plan)
        self.prev_namo_object_names = None

    def check_two_arm_pick_feasibility(self, obj, action):
        motion_planning_region = self.problem_env.get_region_containing(obj)
        goal_robot_xytheta = action['base_pose']

        if self.problem_env.check_base_pose_feasible(goal_robot_xytheta, obj, motion_planning_region):
            motion, status = self.problem_env.get_base_motion_plan(goal_robot_xytheta, motion_planning_region.name)
        else:
            motion = None
            status = 'NoPath'

        return motion, status

    def get_new_fetching_pick_path(self, namo_obj, motion_planning_region_name):
        # if pre-contact, pick_motion consists of:
        #   \tau(self.c_init, self.c_goal) + \tau(self.c_goal, self.fetch_pick_conf)
        # if post-contact, do not change it

        # divide the self.fetch_pick_path into two parts:
        #   before c_goal and after c_goal, from c_goal to self.pick_conf

        post_c_goal_motion = []
        post_c_pidx = np.inf
        for pidx, p in enumerate(self.fetch_pick_path):
            if np.all(p == self.c_goal):
                post_c_pidx=pidx
            if pidx > post_c_pidx:
                post_c_goal_motion.append(p)

        if len(post_c_goal_motion) == 0:
            # self.c_goal is not present in the fetching path
            c_goal = self.fetch_pick_conf
        else:
            c_goal = self.c_goal

        is_fetch_pick_one_arm = self.fetch_pick_op_instance['operator'] == 'one_arm_pick'

        # initial_config = new robot configuration where it placed the object
        # what happens if I go outside of the region in after I gotten into the fetching region?
        # My c_goal should be entrance config? This is complex to code, easier to explain.
        # Just set the shelf region to be some area around the table, and assume you don't have big boxes in
        # that region
        pre_c_goal_motion, status = self.get_motion_plan_with_disabling(c_goal,
                                                                        motion_planning_region_name,
                                                                        is_fetch_pick_one_arm)
        pick_motion = pre_c_goal_motion + post_c_goal_motion
        if status == 'NoPath':
            return None, 'NoPath'

        return pick_motion, status

    def get_rotations_around_z_wrt_gripper(self, object, conf):
        # todo place the object at the designated conf
        self.problem_env.set_arm_base_config(conf)
        T_robot_obj = self.get_robot_transform_wrt_obj(object)

        conf_list = []
        for rotation in [0, np.pi/2, np.pi, 3*np.pi/2]:
            xytheta = get_body_xytheta(object)
            xytheta[0, -1] = xytheta[0, -1] + rotation
            set_obj_xytheta(xytheta, object)
            T_obj_new = object.GetTransform()
            T_robot_new = np.dot(T_obj_new, T_robot_obj)
            self.robot.SetTransform(T_robot_new)
            new_conf = copy.deepcopy(conf)
            new_conf[-3:] = get_body_xytheta(self.robot).squeeze()
            conf_list.append(new_conf)

        return conf_list

    def get_new_fetching_place_path(self, namo_obj, motion_planning_region_name):
        # pick the object
        is_fetch_pick_one_arm = self.fetch_pick_op_instance['operator'] == 'one_arm_pick'
        if is_fetch_pick_one_arm:
            one_arm_pick_object(self.fetching_obj, self.problem_env.robot, self.fetch_pick_op_instance['action'])
        else:
            two_arm_pick_object(self.fetching_obj, self.problem_env.robot, self.fetch_pick_op_instance['action'])

        # get pre_exit_motion -- motion prev to the exit
        pre_exit_motion = []
        for pidx, p in enumerate(self.fetch_place_path):
            pre_exit_motion.append(p)
            if np.all(p == self.fetch_pick_path_exit):
                break

        with self.robot:
            # place it first
            one_arm_place_object(self.fetching_obj, self.robot, self.fetch_place_op_instance['action'])
            place_conf_list = self.get_rotations_around_z_wrt_gripper(self.fetching_obj, self.fetch_place_conf)
            one_arm_pick_object(self.fetching_obj, self.robot, self.fetch_place_op_instance['action'])
            # put it back to where it was
            one_arm_place_object(self.fetching_obj, self.robot, self.fetch_pick_op_instance['action'])
            one_arm_pick_object(self.fetching_obj, self.robot, self.fetch_pick_op_instance['action'])
            place_conf_list = [conf for conf in place_conf_list if self.problem_env.check_base_pose_feasible(conf[-3:], self.fetching_obj, self.problem_env.regions[motion_planning_region_name])]

            if is_fetch_pick_one_arm:
                self.problem_env.set_arm_base_config(self.fetch_pick_path_exit)
            else:
                set_robot_config(self.fetch_pick_path_exit)

            if len(place_conf_list) == 0:
                print "Moved packin region, no feasible base pose"
                return None, "NoPath"

            for conf in place_conf_list:
                # prevent moving the namo_obj = self.packing_region_obj in this case, because you just moved it
                post_exit_motion, status = self.get_motion_plan_with_disabling(conf,
                                                                               motion_planning_region_name,
                                                                               is_fetch_pick_one_arm,
                                                                               exception_obj=namo_obj)
                if status == 'HasSolution':
                    # update the fetch place config
                    self.fetch_place_conf = conf
                    self.fetch_place_op_instance['action']['base_pose'] = conf[-3:]
                    break

        if is_fetch_pick_one_arm:
            one_arm_place_object(self.fetching_obj, self.problem_env.robot, self.fetch_pick_op_instance['action'])
        else:
            two_arm_place_object(self.fetching_obj, self.problem_env.robot, self.fetch_pick_op_instance['action'])

        if status == "NoPath":
            return None, "NoPath"
        else:
            place_motion = pre_exit_motion + post_exit_motion
            return place_motion, status

    def check_two_arm_place_feasibility(self, namo_obj, action, obj_placement_region):
        curr_region = self.problem_env.get_region_containing(self.problem_env.robot)
        if obj_placement_region.name.find('shelf') != -1:
            motion_planning_region_name = 'home_region'
        else:
            motion_planning_region_name = obj_placement_region.name
        print motion_planning_region_name

        goal_robot_xytheta = action['base_pose']
        pick_base_pose = get_body_xytheta(self.problem_env.robot)
        pick_conf = self.problem_env.robot.GetDOFValues()

        current_collisions = self.curr_namo_object_names
        self.prev_namo_object_names = current_collisions

        namo_status = 'NoPath'
        namo_place_motion = None
        if self.problem_env.check_base_pose_feasible(goal_robot_xytheta, namo_obj,
                                                     self.problem_env.regions[motion_planning_region_name]):
            namo_place_motion, namo_status = self.problem_env.get_base_motion_plan(goal_robot_xytheta,
                                                                                    motion_planning_region_name)
        if namo_status == 'NoPath':
            return namo_place_motion, namo_status, self.curr_namo_object_names

        two_arm_place_object(namo_obj, self.problem_env.robot, action)
        if self.is_c_init_pre_contact:
            fetch_pick_path, fetching_pick_status = self.get_new_fetching_pick_path(namo_obj,
                                                                                         motion_planning_region_name)
            if fetching_pick_status == "NoPath":
                return None, 'NoPath', self.curr_namo_object_names
        else:
            fetch_pick_path = self.fetch_pick_path

        packing_region_moved = self.packing_region_obj == namo_obj # note this can only happen in the pre_contact stage
        if packing_region_moved:
            self.update_target_namo_place_base_pose()
            fetch_place_path, fetching_place_status = self.get_new_fetching_place_path(namo_obj,
                                                                                       motion_planning_region_name)
            if fetching_place_status == "NoPath":
                return None, 'NoPath', self.curr_namo_object_names
        else:
            fetch_place_path = self.fetch_place_path

        new_collisions = self.get_obstacles_in_collision_from_fetching_path(fetch_pick_path, fetch_place_path)
        new_collisions = [tmp for tmp in new_collisions if self.problem_env.get_region_containing(tmp) == self.curr_namo_region]
        import pdb;pdb.set_trace()

        # go back to pre-place
        self.problem_env.robot.SetDOFValues(pick_conf)
        set_robot_config(action['base_pose'], self.robot)
        grab_obj(self.problem_env.robot, namo_obj)
        set_robot_config(pick_base_pose, self.robot)

        # if new collisions is more than or equal to the current collisions, don't bother executing it
        if len(current_collisions) <= len(new_collisions):
            return None, "NoPath", self.curr_namo_object_names

        # otherwise, update the new namo objects
        self.curr_namo_object_names = [obj.GetName() for obj in new_collisions]

        # pick motion is the path to the fetching object, place motion is the path to place the namo object
        motion = {'fetch_pick_path': fetch_pick_path, 'fetch_place_path': fetch_place_path,
                  'place_motion': namo_place_motion}

        # update the self.fetch_place_path if the packing region has moved
        self.fetch_pick_path = fetch_pick_path
        self.fetch_place_path = fetch_place_path

        # todo: change the place entrance configuration too

        # update the high level controller task plan
        namo_region = self.problem_env.get_region_containing(self.fetching_obj)
        self.high_level_controller.set_task_plan([{'region': namo_region, 'objects': new_collisions}])

        return motion, "HasSolution", self.curr_namo_object_names

    def check_one_arm_pick_feasibility(self, obj, action):
        base_pose = action['base_pose']
        g_config = action['g_config']
        if g_config is None:
            return None, "NoPath"

        curr_obj_region = self.problem_env.get_region_containing(obj)
        curr_base_region = self.problem_env.get_region_containing(self.robot)
        if curr_obj_region.name.find('shelf') != -1:
            target_base_region = curr_base_region
        else:
            target_base_region = curr_obj_region

        if not self.problem_env.check_base_pose_feasible(base_pose, obj, target_base_region):
            return None, "NoPath"

        pick_config = np.hstack([g_config, base_pose.squeeze()])
        obj.Enable(False)  # note this is because the pick_grasp is in collision with target
        pick_motion, status = self.problem_env.get_arm_base_motion_plan(pick_config)
        obj.Enable(True)

        return pick_motion, status

    def check_one_arm_place_feasibility(self, namo_obj, action, obj_placement_region):
        g_config = action['g_config']
        if g_config is None:
            return None, 'NoPath'
        base_pose = action['base_pose']
        motion_planning_region = self.problem_env.get_region_containing(self.problem_env.robot)
        full_place_config = np.hstack([g_config, base_pose.squeeze()])

        current_collisions = self.curr_namo_object_names
        self.prev_namo_object_names = current_collisions

        # check if place motion exists for the namo object
        namo_place_motion, status = self.problem_env.get_arm_base_motion_plan(full_place_config,
                                                                              motion_planning_region.name)

        if status == 'NoPath':
            return namo_place_motion, "NoPath", self.curr_namo_object_names

        one_arm_place_object(namo_obj, self.robot, action) # don't fold your arm?

        if self.is_c_init_pre_contact:
            fetch_pick_path, fetching_pick_status = self.get_new_fetching_pick_path(namo_obj, motion_planning_region.name)
            if fetching_pick_status == "NoPath":
                # todo does it not work with the gripper opened?
                import pdb;pdb.set_trace()
                return None, 'NoPath', self.curr_namo_object_names
        else:
            fetch_pick_path = self.fetch_pick_path

        fetch_place_path = self.fetch_place_path

        new_collisions = self.get_obstacles_in_collision_from_fetching_path(fetch_pick_path, fetch_place_path)
        new_collisions = [tmp for tmp in new_collisions if
                          self.problem_env.get_region_containing(tmp) == self.curr_namo_region]

        one_arm_pick_object(namo_obj, self.robot, action)

        import pdb;pdb.set_trace()
        # if new collisions is more than or equal to the current collisions, don't bother executing it
        if len(current_collisions) <= len(new_collisions):
            return None, "NoPath", self.curr_namo_object_names

        # otherwise, update the new namo objects
        self.curr_namo_object_names = [obj.GetName() for obj in new_collisions]

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

    def get_obstacles_in_collision_from_fetching_path(self, pick_path, place_path):
        curr_region = self.problem_env.get_region_containing(self.problem_env.robot)
        is_fetch_pick_one_arm = self.fetch_pick_op_instance['operator'] == 'one_arm_pick'
        is_fetch_place_one_arm = self.fetch_place_op_instance['operator'] == 'one_arm_place'

        assert len(self.robot.GetGrabbed()) == 0, 'this function assumes the robot starts with object in hand'

        ### fetch place path collisions
        if is_fetch_pick_one_arm:
            one_arm_pick_object(self.fetching_obj, self.problem_env.robot, self.fetch_pick_op_instance['action'])
        else:
            two_arm_pick_object(self.fetching_obj, self.problem_env.robot, self.fetch_pick_op_instance['action'])

        if is_fetch_place_one_arm:
            manip = self.problem_env.robot.GetManipulator('rightarm_torso')
            self.problem_env.robot.SetActiveDOFs(manip.GetArmIndices(),
                                                 DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
        else:
            self.problem_env.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

        # this really stays the same if we do not move the packing region
        place_collisions = self.problem_env.get_objs_in_collision(place_path, curr_region.name)

        ### fetch pick path collisions
        if is_fetch_place_one_arm:
            one_arm_place_object(self.fetching_obj, self.robot, self.fetch_pick_op_instance['action'])
        else:
            two_arm_place_object(self.fetching_obj, self.robot, self.fetch_pick_op_instance['action'])

        if is_fetch_pick_one_arm:
            manip = self.problem_env.robot.GetManipulator('rightarm_torso')
            self.problem_env.robot.SetActiveDOFs(manip.GetArmIndices(), DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
        else:
            self.problem_env.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
        pick_collisions = self.problem_env.get_objs_in_collision(pick_path, curr_region.name)

        collisions = pick_collisions
        collisions += [p for p in place_collisions if p not in collisions]

        # go back to the original robot state
        if is_fetch_pick_one_arm:
            one_arm_place_object(self.fetching_obj, self.problem_env.robot, self.fetch_pick_op_instance['action'])
        else:
            two_arm_place_object(self.fetching_obj, self.problem_env.robot, self.fetch_pick_op_instance['action'])

        return collisions

    def solve_single_object(self, c_init, c_goal, curr_namo_region, initial_collision_names, mcts):
        # get the initial collisions
        self.curr_namo_object_names = [tmp for tmp in initial_collision_names
                                       if self.problem_env.get_region_containing(self.env.GetKinBody(tmp)) == curr_namo_region]
        self.init_namo_object_names = copy.deepcopy(self.curr_namo_object_names)

        namo_problem_for_obj = [{'region': curr_namo_region, 'objects': [self.env.GetKinBody(obj_name)
                                                                    for obj_name in self.curr_namo_object_names]}]
        self.high_level_controller.set_task_plan(namo_problem_for_obj)
        mcts.update_init_node_obj()

        self.problem_env.high_level_planner = self.high_level_controller
        self.is_c_init_pre_contact = len([tmp for tmp in self.fetch_pick_path if np.all(tmp == c_init)]) > 0
        self.c_goal = c_goal
        self.curr_namo_region = curr_namo_region

        # setup the task plan
        prenamo_initial_node = mcts.s0_node
        namo_plan = None
        goal_node = None
        self.problem_env.namo_planner = self

        while namo_plan is None:
            import pdb;pdb.set_trace()
            mcts.switch_init_node(prenamo_initial_node)
            self.problem_env.is_solving_namo = True
            search_time_to_reward, namo_plan, goal_node = mcts.search(n_iter=self.n_iter,
                                                                      n_optimal_iter=self.n_optimal_iter)
            self.problem_env.is_solving_namo = False

        if namo_plan is not None:
            self.problem_env.apply_plan(namo_plan)
            goal_node.state_saver = DynamicEnvironmentStateSaver(self.problem_env.env)
        else:
            mcts.switch_init_node(prenamo_initial_node)

        return namo_plan, goal_node

    def get_robot_transform_wrt_obj(self, target_obj):
        T_target_obj = target_obj.GetTransform()
        T_robot_world = self.problem_env.robot.GetTransform()
        # solve T_packing_region_world * T_robot_obj = T_robot_world
        T_robot_obj = np.linalg.solve(T_target_obj, T_robot_world)
        return T_robot_obj

    def get_world_robot_transform_given_obj_transform(self, target_obj):
        T_robot_world_prime = np.dot(target_obj.GetTransform(), self.T_robot_obj)
        return T_robot_world_prime

    def update_target_namo_place_base_pose(self):
        T_robot = self.get_world_robot_transform_given_obj_transform(self.packing_region_obj)
        with self.robot:
            self.robot.SetTransform(T_robot)
            fetch_place_base_pose = get_body_xytheta(self.robot)
            self.fetch_place_conf[-3:] = fetch_place_base_pose
            self.fetch_place_op_instance['action']['base_pose'] = fetch_place_base_pose

    def set_arm_base_config(self, config):
        manip = self.problem_env.robot.GetManipulator('rightarm_torso')
        self.problem_env.robot.SetActiveDOFs(manip.GetArmIndices(),
                                             DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
        set_active_dof_conf(config, self.robot)

    def initialize_namo_problem(self, fetch_plan, target_obj, fetch_goal_node,
                                fetch_pick_path_exit, fetch_place_path_entrance, target_packing_region_name):
        self.packing_region_obj = None
        self.is_movable_packing_region = target_packing_region_name.find('packing_box') != -1
        self.fetching_obj = target_obj

        self.fetch_pick_op_instance = fetch_plan[0]
        self.fetch_place_op_instance = fetch_plan[1]
        self.fetch_pick_path = fetch_plan[0]['path']
        self.fetch_place_path = fetch_plan[1]['path']
        self.fetch_pick_path_exit = fetch_pick_path_exit
        self.fetch_place_path_entrance = fetch_place_path_entrance

        self.fetch_pick_conf = self.problem_env.make_config_from_op_instance(self.fetch_pick_op_instance)
        self.fetch_place_conf = self.problem_env.make_config_from_op_instance(self.fetch_place_op_instance)
        self.problem_env.high_level_planner = self.high_level_controller

        if self.is_movable_packing_region:
            # getting relative robot base config wrt the packing region object
            with self.robot:
                self.set_arm_base_config(self.fetch_place_conf)
                self.packing_region_obj = self.problem_env.env.GetKinBody(target_packing_region_name)
                self.T_robot_obj = self.get_robot_transform_wrt_obj(self.packing_region_obj)

        self.target_namo_place_base_pose = self.fetch_place_conf

        self.fetch_goal_node = fetch_goal_node
        self.prefetching_robot_config = get_body_xytheta(self.problem_env.robot).squeeze()

    ########################################################################################

