import sys


## NAMO problem environment
from namo_env import NAMO

## mover library utility functions
sys.path.append('../mover_library/')
from utils import set_robot_config, get_body_xytheta, check_collision_except,  grab_obj, \
    simulate_path, two_arm_pick_object, two_arm_place_object

from operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp
OBJECT_ORIGINAL_COLOR = (0, 0, 0)
COLLIDING_OBJ_COLOR = (0, 1, 1)
TARGET_OBJ_COLOR = (1, 0, 0)


class StripStreamNAMO(NAMO):
    def __init__(self):
        NAMO.__init__(self)
        self.init_saver = self.problem_config['initial_saver']

    def apply_two_arm_pick_action(self, action, obj):
        pick_base_pose, grasp_params = action
        set_robot_config(pick_base_pose, self.robot)
        grasps = compute_two_arm_grasp(depth_portion=grasp_params[2],
                                       height_portion=grasp_params[1],
                                       theta=grasp_params[0],
                                       obj=obj,
                                       robot=self.robot)
        g_config = solveTwoArmIKs(self.env, self.robot, obj, grasps)

        action = {}
        action['g_config'] = g_config
        action['base_pose'] = pick_base_pose
        two_arm_pick_object(obj, self.robot, action)

    def apply_two_arm_place_action(self, action, node, check_feasibility, parent_motion):
        pass











