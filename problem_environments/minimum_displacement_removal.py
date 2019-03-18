import numpy as np

from openravepy import DOFAffine
from planners.mcts_utils import make_action_hashable
from manipulation.primitives.savers import DynamicEnvironmentStateSaver
from trajectory_representation.operator import Operator

from problem_environment import ProblemEnvironment
from mover_problem import MoverProblem

## mover library utility functions
from mover_library.utils import set_robot_config, grab_obj, two_arm_pick_object, two_arm_place_object, \
    get_trajectory_length

OBJECT_ORIGINAL_COLOR = (0, 0, 0)
COLLIDING_OBJ_COLOR = (0, 1, 1)
TARGET_OBJ_COLOR = (1, 0, 0)


class MinimumDisplacementRemoval(ProblemEnvironment):
    def __init__(self, problem_idx):
        ProblemEnvironment.__init__(self)
        problem = MoverProblem(self.env, problem_idx)
        self.problem_config = problem.get_problem_config()
        self.robot = self.env.GetRobots()[0]
        self.objects = self.problem_config['objects']
        self.regions = {'entire_region': self.problem_config['entire_region']}
        self.init_base_conf = self.problem_config['init_base_config']
        self.goal_base_conf = self.problem_config['goal_base_config']
        self.problem_idx = self.problem_config['problem_idx']

        self.init_saver = DynamicEnvironmentStateSaver(self.env)
        self.robot = self.env.GetRobots()[0]
        self.objects = self.problem_config['objects']
        self.regions = {'entire_region': self.problem_config['entire_region']}
        self.infeasible_reward = -2
        self.is_init_pick_node = True

        self.init_saver = DynamicEnvironmentStateSaver(self.env)
        self.problem_config['env'] = self.env
        self.objects_not_in_goal = []

    def set_objects_not_in_goal(self, objects_not_in_goal):
        self.objects_not_in_goal = objects_not_in_goal

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
                    if len(self.namo_planner.prev_namo_object_names) - len(new_namo_obj_names) > 0:
                        len(self.namo_planner.fixed_init_namo_object_names) - len(new_namo_obj_names)
                        distance_travelled = get_trajectory_length(motion_plan['place_motion'])
                        reward = min(1/distance_travelled, 2)
                    else:
                        distance_travelled = get_trajectory_length(motion_plan['place_motion'])
                        reward = max(-distance_travelled, self.infeasible_reward)
                    objs_in_collision = [self.env.GetKinBody(name) for name in new_namo_obj_names]
                else:
                    objs_in_collision = [self.env.GetKinBody(name) for name in self.namo_planner.curr_namo_object_names]
                    reward = 0
        else:
            reward = self.infeasible_reward

        return reward, objs_in_collision

    def reset_to_init_state(self, node):
        saver = node.state_saver
        saver.Restore()  # this call re-enables objects that are disabled
        self.curr_state = self.get_state()
        self.objects_not_in_goal = node.objects_not_in_goal

        if not self.init_which_opreator != 'two_arm_pick':
            grab_obj(self.robot, self.curr_obj)

        is_parent_action_pick = node.parent_action['operator_name'].find('pick') != -1 \
                                    if node.parent_action is not None else False
        if is_parent_action_pick:
            two_arm_pick_object(node.parent.obj, self.robot, node.parent_action)

        if self.is_solving_namo:
            self.namo_planner.reset()
        #self.high_level_planner.reset_task_plan_indices()

        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    def set_init_namo_object_names(self, object_names):
        self.namo_planner.init_namo_object_names = object_names

    def check_base_pose_feasible(self, base_pose, obj, region):
        if base_pose is None:
            return False
        if not self.is_collision_at_base_pose(base_pose, obj) \
                and self.is_in_region_at_base_pose(base_pose, obj, robot_region=region,
                                                   obj_region=region):
                return True
        return False

    def apply_two_arm_pick_action(self, action, node, check_feasibility, parent_motion):
        obj = node.obj
        region = node.region

        motion_plan = None
        status = "NoSolution"

        if check_feasibility:
            if self.is_solving_fetching:
                motion_plan, status = self.fetch_planner.check_two_arm_pick_feasibility(obj, action, region)
            elif self.is_solving_namo:
                motion_plan, status = self.namo_planner.check_two_arm_pick_feasibility(obj, action)
        else:
            motion_plan = parent_motion
            status = 'HasSolution'
            #reward = node.parent_action_reward
            #objs_in_collision = node.objs_in_collision

        reward, objs_in_collision = self.determine_reward('two_arm_pick', obj, motion_plan, status)
        if status == 'HasSolution':
            try:
                two_arm_pick_object(obj, self.robot, action)
                curr_state = self.get_state()
            except:
                import pdb;pdb.set_trace()
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
                try:
                    self.namo_planner.prev_namo_object_names = [namo_obj.GetName() for namo_obj in node.parent.objs_in_collision]
                    self.namo_planner.curr_namo_object_names = [namo_obj.GetName() for namo_obj in new_namo_objs]
                except:
                    import pdb;pdb.set_trace()

                #self.namo_planner.fetch_pick_path = node.children[make_action_hashable(action)].parent_motion['fetch_pick_path']
                #self.namo_planner.fetch_place_path = node.children[make_action_hashable(action)].parent_motion['fetch_place_path']

                # todo update the task-plan?
                #self.high_level_planner.set_task_plan([{'region': self.regions['entire_region'], 'objects': new_namo_objs}])
        reward, objs_in_collision = self.determine_reward('two_arm_place', target_obj, plan, status, new_namo_obj_names)
        #if reward > 1:
        #    import pdb;pdb.set_trace()

        if status == 'HasSolution':
            two_arm_place_object(target_obj, self.robot, action)
            curr_state = self.get_state()
            #if self.is_solving_namo:
            #    set_robot_config(self.namo_planner.current_pick_conf, self.robot)
        else:
            curr_state = self.get_state()

        return curr_state, reward, plan, objs_in_collision

    def is_goal_reached(self):
        return len(self.objects_not_in_goal) == 0

    def which_operator(self):
        if self.is_pick_time():
            return 'two_arm_pick'
        else:
            return 'two_arm_place'

    def apply_action(self, action):

        import pdb;pdb.set_trace()
        reward = 1
        objects_not_in_goal = 0
        return reward, objects_not_in_goal

    def get_place_region_for_object(self, object):
        return 0

    def get_applicable_op(self):
        op_name = self.which_operator()
        if op_name == 'two_arm_place':
            op = Operator(operator_type=op_name,
                          discrete_parameters={'region': self.regions['entire_region']},
                          continuous_parameters=None,
                          low_level_motion=None)
        else:
            op = Operator(operator_type=op_name,
                          discrete_parameters={'object': self.objects_not_in_goal[0]},
                          continuous_parameters=None,
                          low_level_motion=None)
        return op








