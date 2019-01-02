import sys
import numpy as np
from manipulation.bodies.bodies import set_config

sys.path.append('../mover_library/')
from samplers import sample_pick, sample_grasp_parameters, sample_ir, sample_ir_multiple_regions, sample_one_arm_grasp_parameters

from utils import compute_occ_vec, set_robot_config, remove_drawn_configs, \
    draw_configs, clean_pose_data, draw_robot_at_conf, \
    check_collision_except, one_arm_pick_object, release_obj, one_arm_place_object, set_config

sys.path.append('../mover_library/')
from operator_utils.grasp_utils import solveIK, compute_one_arm_grasp
from openravepy import IkFilterOptions, IkReturnAction

def check_collision_except_obj(obj, robot, env):
    in_collision = (check_collision_except(obj, robot, env)) \
                   or (check_collision_except(robot, obj, env))
    return in_collision


class PickUnif(object):
    def __init__(self, problem_env):
        self.problem_env = problem_env
        self.env = problem_env.env
        self.robot = self.env.GetRobots()[0]

    def predict(self, obj, region):
        raise NotImplementedError


class OneArmPickUnif(PickUnif):
    def __init__(self, env):
        PickUnif.__init__(self, env)

    def compute_grasp_action(self, obj, region, n_iter=1000):
        rightarm_torso_manip = self.robot.GetManipulator('rightarm_torso')

        for iter in range(n_iter):
            pick_base_pose = None
            print 'Sampling IR...'
            while pick_base_pose is None:
                with self.robot:
                    pick_base_pose = sample_ir(obj, self.robot, self.env, region, n_iter=1)
            print 'Done!'
            theta, height_portion, depth_portion = sample_one_arm_grasp_parameters()
            grasp_params = np.array([theta[0], height_portion[0], depth_portion[0]])

            with self.robot:
                set_robot_config(pick_base_pose, self.robot)
                grasps = compute_one_arm_grasp(depth_portion=grasp_params[2],
                                               height_portion=grasp_params[1],
                                               theta=grasp_params[0],
                                               obj=obj,
                                               robot=self.robot)
                for g in grasps:
                    g_config = rightarm_torso_manip.FindIKSolution(g, 0)
                    if g_config is not None:
                        set_config(self.robot, g_config, self.robot.GetManipulator('rightarm_torso').GetArmIndices())
                        # Todo
                        #   I might be able to turn to disabling obstacles quickly if the collision, not the base pose,
                        #   is the problem
                        if not self.env.CheckCollision(self.robot): #check_collision_except(obj, self.env):
                            pick_params = {'operator_name': 'one_arm_pick', 'base_pose': pick_base_pose, 'grasp_params': grasp_params, 'g_config':g_config}
                            return pick_params
                        #one_arm_place_obj(obj, self.robot)

        print "Sampling one arm pick failed"
        pick_params = {'operator_name': 'one_arm_pick', 'base_pose': None, 'grasp_params': None, 'g_config': None}
        return pick_params

    def predict(self, obj, region):
        print "Sampling one arm pick..."
        if self.problem_env.is_solving_namo:
            pick_params = self.compute_grasp_action(obj, region, n_iter=100)
        else:
            pick_params = self.compute_grasp_action(obj, region, n_iter=0)

        # pick_params = {}
        # pick_params['g_config'] = None
        if self.problem_env.is_solving_fetching and pick_params['g_config'] is None:
            self.problem_env.disable_objects_in_region(region.name)
            obj.Enable(True)
            print "Disabled the objects, computing one arm pick..."
            pick_params = self.compute_grasp_action(obj, region, n_iter=100)
            self.problem_env.enable_objects_in_region(region.name)
        # if pick_params['g_config'] is None:
        #    import pdb;pdb.set_trace()
        print pick_params.keys()
        return pick_params


