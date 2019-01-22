import numpy as np
import sys
import copy
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../mover_library/')
from conveyor_belt_problem import create_conveyor_belt_problem
from problem_environment import ProblemEnvironment
import cPickle as pickle

from mover_library.utils import *
from operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp
from manipulation.primitives.savers import DynamicEnvironmentStateSaver


class ConveyorBelt(ProblemEnvironment):
    def __init__(self, problem_idx):
        self.problem_idx = problem_idx
        ProblemEnvironment.__init__(self)
        obj_setup = self.load_object_setup()
        self.problem_config = create_conveyor_belt_problem(self.env, obj_setup)
        self.objects = self.problem_config['objects']
        self.init_base_conf = np.array([0, 1.05, 0])
        self.fetch_planner = None

        #self.robot_region = self.problem_config['entire_region']
        #self.obj_region = self.problem_config['loading_region']

        self.regions = {'entire_region': self.problem_config['entire_region'],
                        'object_region': self.problem_config['loading_region']}
        self.robot = self.problem_config['env'].GetRobots()[0]
        self.infeasible_reward = -2
        self.optimal_score = 5

        self.curr_obj = self.objects[0]

        self.curr_state = self.get_state()
        self.objs_to_move = self.objects

        self.init_saver = DynamicEnvironmentStateSaver(self.env)
        self.is_init_pick_node = True
        self.init_obj = self.objects[0]
        self.init_operator = 'two_arm_place'
        self.name = 'convbelt'

    def load_object_setup(self):
        obj_setup = pickle.load(open('./problem_environments/conveyor_belt_domain_problems/' + str(self.problem_idx) + '.pkl', 'r'))
        return obj_setup

    def save_object_setup(self):
        object_configs = {'object_poses': self.problem_config['obj_poses'],
                          'object_shapes': self.problem_config['obj_shapes'],
                          'obst_poses': self.problem_config['obst_poses'],
                          'obst_shapes': self.problem_config['obst_shapes']}
        pickle.dump(object_configs, open('./problem_environments/conveyor_belt_domain_problems/' + str(self.problem_idx) + '.pkl', 'wb'))

    def apply_two_arm_pick_action_stripstream(self, action, obj=None, do_check_reachability=False):
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
        try:
            assert g_config is not None
        except:
            import pdb;pdb.set_trace()

        action = {'base_pose': pick_base_pose, 'g_config': g_config}
        two_arm_pick_object(obj_to_pick, self.robot, action)
        set_robot_config(self.init_base_conf, self.robot)
        curr_state = self.get_state()
        reward = 0
        pick_path = None
        return curr_state, reward, g_config, pick_path

    def update_next_obj_to_pick(self, place_action):
        self.placements.append(place_action)
        if len(self.placements) < len(self.objects):
            self.curr_obj = self.objs_to_move[len(self.placements)]  # update the next object to be picked

    def apply_two_arm_pick_action(self, action, node, check_feasibility, parent_motion):
        if action['g_config'] is None:
            curr_state = self.get_state()
            return curr_state, self.infeasible_reward, None, []

        object_to_pick = node.obj
        if check_feasibility:
            two_arm_pick_object(object_to_pick, self.robot, action)
            set_robot_config(self.init_base_conf, self.robot)
            if self.env.CheckCollision(self.robot):
                two_arm_place_object(object_to_pick, self.robot, action)
                set_robot_config(self.init_base_conf, self.robot)
                curr_state = self.get_state()
                return curr_state, self.infeasible_reward, None, []
        else:
            g_config = parent_motion
        two_arm_pick_object(object_to_pick, self.robot, action)
        set_robot_config(self.init_base_conf, self.robot)
        curr_state = self.get_state()
        reward = 0
        return curr_state, reward, action['g_config'], []

    def apply_two_arm_place_action(self, action, node, check_feasibility, parent_motion):
        if action['base_pose'] is None:
            curr_state = self.get_state()
            return curr_state, self.infeasible_reward, None, []

        target_obj = node.obj
        target_region = node.region
        place_base_pose = action['base_pose']
        if check_feasibility:
            path, status = self.get_base_motion_plan(place_base_pose.squeeze())
        else:
            path = parent_motion
            status = "HasSolution"

        if status == 'HasSolution':
            two_arm_place_object(target_obj, self.robot, action)
            set_robot_config(self.init_base_conf, self.robot)
            curr_state = self.get_state()
            reward = 1
            return curr_state, reward, path, []
        else:
            curr_state = self.get_state()
            return curr_state, self.infeasible_reward, None, []

    def reset_to_init_state_stripstream(self):
        self.init_saver.Restore()
        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    def reset_to_init_state(self, node):
        saver = node.state_saver
        saver.Restore()  # this call re-enables objects that are disabled
        self.curr_state = self.get_state()
        self.placements = copy.deepcopy(self.initial_placements)

        self.curr_obj = self.objs_to_move[len(self.placements)] # todo change it for NAMO domain
        if self.init_operator != 'two_arm_pick':
            grab_obj(self.robot, self.curr_obj)
        self.high_level_planner.reset_task_plan_indices()
        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    def is_goal_reached(self):
        return len(self.placements) == 5

    def set_init_state(self, saver):
        self.init_saver = saver
        self.initial_placements = copy.deepcopy(saver.placements)
        # I think you should restore first?
        self.init_which_opreator = saver.which_operator
        self.objs_to_move = self.objects[len(saver.placements):]

    def disable_objects(self):
        for o in self.objects:
            o.Enable(False)

    def enable_objects(self):
        for o in self.objects:
            o.Enable(True)

    def which_operator(self, obj=None):
        if self.is_pick_time():
            return 'two_arm_pick'
        else:
            return 'two_arm_place'


