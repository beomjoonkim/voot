from planners.mcts import MCTS

from planners.mcts import MCTS
from generators.PlaceUniform import PlaceUnif
from generators.PickUniform import PickWithBaseUnif

from sampling_strategies.uniform import Uniform
from manipulation.primitives.savers import DynamicEnvironmentStateSaver

import sys
sys.path.append('../mover_library/')

from utils import get_body_xytheta, visualize_path, release_obj, set_robot_config, grab_obj


class ConstraintRemovalPlanner:
    def __init__(self, problem_env):
        self.problem_env = problem_env
        self.robot = self.problem_env.robot
        self.pick_motion = None
        self.pick_config = None
        self.pick_region_name = None
        self.place_motion = None
        self.is_pick_motion_collision = False
        self.prepick_state_saver = None
        self.prepick_config = None

    def reset_to_prepick(self):
        #assert len(self.robot.GetGrabbed()) == 0
        self.prepick_state_saver.Restore()

    def initiate_crp_problem(self, target_obj, action, region_name):
        self.pick_region_name = region_name
        self.pick_config, self.pick_grasp = action
        self.prepick_config = get_body_xytheta(self.robot)
        self.prepick_state_saver = DynamicEnvironmentStateSaver(self.problem_env.env)
        self.namo_target_obj = target_obj

    def compute_pick_motion_for_target_obj(self):
        motion, status = self.problem_env.get_motion_plan(self.pick_config, self.pick_region_name)
        is_feasible_without_constraint_removal = status == 'HasSolution'
        if is_feasible_without_constraint_removal:
            self.pick_motion = motion
            self.is_pick_motion_collision = False
        else:
            self.problem_env.disable_objects_in_region(self.pick_region_name)
            self.namo_target_obj.Enable(True)

            # todo enable the current target object
            self.pick_motion, status = self.problem_env.get_motion_plan(self.pick_config, self.pick_region_name)
            self.problem_env.enable_objects_in_region(self.pick_region_name)
            if status == 'NoPath':
                return None
            else:
                self.is_pick_motion_collision = True
        return self.pick_motion

    def pick_namo_target_obj(self):
        set_robot_config(self.pick_config, self.robot)
        self.robot.SetDOFValues(self.pick_full_conf)
        grab_obj(self.robot, self.namo_target_obj)

    def release_namo_target_obj(self):
        self.problem_env.place_object(self.pick_config)

    def make_constraint_removal_plan(self):
        # from the original problem, this function is called once the robot has the object in hand
        # from the namo problem, this function is called once the robot has moved an object out of the way
        was_target_obj_held = len(self.robot.GetGrabbed()) != 0 and self.robot.GetGrabbed()[0] == self.namo_target_obj

        with self.robot:
            self.pick_namo_target_obj()
            place_motion_collisions = self.problem_env.get_objs_in_collision(self.place_motion, self.pick_region_name)

            if self.is_pick_motion_collision:
                self.problem_env.place_object(self.pick_config)
                pick_motion_collisions = self.problem_env.get_objs_in_collision(self.pick_motion, self.pick_region_name)
                self.pick_namo_target_obj() # if this was held, then pick it up again
            else:
                pick_motion_collisions = []
            if not was_target_obj_held:
                self.release_namo_target_obj()

        # todo do not add duplicates
        #collisions = place_motion_collisions + [p for p in pick_motion_collisions if p not in place_motion_collisions]
        collisions = [p for p in pick_motion_collisions if p not in place_motion_collisions] + place_motion_collisions
        return collisions

    def update_next_namo_obj(self):
        objs_in_collision = self.make_constraint_removal_plan()
        self.problem_env.curr_namo_object_names = [o.GetName() for o in objs_in_collision]

    def is_last_obj(self, obj):
        return obj.GetName() == self.namo_object_names[-1]

    def add_picking_target_obj_to_plan(self, plan):
        last_step = plan[-1]
        last_place_path = last_step['path']['path_to_last_place']
        path_to_target = last_step['path']['path_to_pick_target_obj']

        plan[-1]['path'] = last_place_path
        picking_target_step = {'action': (self.pick_config, self.pick_grasp),
                               'operator': 'two_arm_pick',
                               'path': (self.pick_full_conf, path_to_target),
                               'obj_name': self.namo_target_obj.GetName()}
        plan.append(picking_target_step)
        return plan

    def add_placing_target_obj_to_plan(self, plan):
        placing_target_step = {'action': (self.place_config),
                               'operator': 'two_arm_place',
                               'path': self.place_motion,
                               'obj_name': self.namo_target_obj.GetName()}
        plan.append(placing_target_step)
        return plan

    def compute_constraint_removal_motion(self, place_config, place_region_name):
        self.place_motion, status = self.problem_env.get_motion_plan(place_config, place_region_name)
        assert self.robot.GetGrabbed() != 0, 'CRP must be solved once the target obj is held'

        self.place_region = place_region_name
        self.place_config = place_config
        self.pick_full_conf = self.robot.GetDOFValues()

        if status == 'NoPath':
            self.problem_env.disable_objects_in_region(self.pick_region_name)
            held_obj = self.robot.GetGrabbed()[0]
            held_obj.Enable(True)
            self.place_motion, _ = self.problem_env.get_motion_plan(place_config, 'entire_region')
            self.problem_env.enable_objects_in_region(self.pick_region_name)
            # this might plan a path that does not collide with any
        else:
            return self.place_motion, 'HasSimpleSolution'

        if self.place_motion is None:
            set_robot_config(self.prepick_config, self.robot)
            self.reset_to_prepick()
            return None, "NoPath"

        objs_in_collision = self.make_constraint_removal_plan()
        pick_region = self.problem_env.regions[self.pick_region_name]
        task_plan = [{'region': pick_region, 'objects': objs_in_collision}]
        if len(objs_in_collision) == 0:
            return self.place_motion, 'HasSimpleSolution'

        uct_parameter = 0
        widening_parameter = 0.5

        # todo set the environment to the configuration before the pick node
        # set the robot to its init_base_pose
        self.problem_env.is_solving_namo = True
        self.problem_env.init_namo_object_names = [o.GetName() for o in objs_in_collision]
        self.problem_env.curr_namo_object_names = [o.GetName() for o in objs_in_collision]
        self.problem_env.namo_pick_path = self.pick_motion
        self.problem_env.namo_place_path = self.place_motion
        self.problem_env.namo_region = self.pick_region_name

        # resetting the robot to its pre-pick configuration
        try:
            self.problem_env.place_object(self.pick_config)
        except:
            import pdb;pdb.set_trace()
        set_robot_config(self.prepick_config, self.robot)
        self.reset_to_prepick()
        two_arm_pick_pi = PickWithBaseUnif(self.problem_env.env, self.robot)
        two_arm_place_pi = PlaceUnif(self.problem_env.env, self.robot)
        sampling_strategy = Uniform(self.problem_env, two_arm_pick_pi, two_arm_place_pi)
        mcts = MCTS(widening_parameter, uct_parameter, sampling_strategy, self.problem_env, 'mover', task_plan)

        #if self.namo_target_obj.GetName() == 'rectangular_packing_box2':
        #    import pdb;pdb.set_trace()
        search_time_to_reward, plan, optimal_score_achieved = mcts.search(n_iter=20)
        #if self.namo_target_obj.GetName() == 'rectangular_packing_box2':
        #    import pdb;pdb.set_trace()

        self.problem_env.is_solving_namo = False
        self.problem_env.namo_object_names = []
        self.problem_env.namo_pick_path = None
        self.problem_env.namo_place_path = None
        self.problem_env.namo_region = None

        # in either case, set the environment to the configuration just after the pick
        if plan is None:
            return None, 'NoPath'
        else:
            try:
                # todo how come it didn't add path_to_pick_target_obj and path_to_last_place?
                plan = self.add_picking_target_obj_to_plan(plan)
                plan = self.add_placing_target_obj_to_plan(plan)
            except:
                import pdb;pdb.set_trace()
            return plan, 'HasSolution'



