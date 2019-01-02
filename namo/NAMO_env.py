##TODO: Clean this
from manipulation.constants import PARALLEL_LEFT_ARM, REST_LEFT_ARM, HOLDING_LEFT_ARM, FOLDED_LEFT_ARM, \
    FAR_HOLDING_LEFT_ARM, LOWER_TOP_HOLDING_LEFT_ARM, REGION_Z_OFFSET
from manipulation.primitives.utils import mirror_arm_config
import numpy as np
import sys
import socket
from NAMO_problem import NAMO_problem

sys.path.append('../mover_library/')
from samplers import *
from utils import *
from operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp
from data_preprocess.preprocessing_utils import compute_fetch_vec
from TreeNode import *
from data_load_utils import convert_collision_vec_to_one_hot
from motion_planner import collision_fn, base_extend_fn, base_sample_fn, base_distance_fn, smooth_path, rrt_connect
import random

OBJECT_ORIGINAL_COLOR = (0, 0, 0)
COLLIDING_OBJ_COLOR = (0, 1, 1)
TARGET_OBJ_COLOR = (1, 0, 0)


class NAMO:
    def __init__(self, manual_pinst=None, is_preprocess=False):
        self.manual_pinst = manual_pinst
        if not is_preprocess:
            manual_pinst = self.get_p_inst()
        # print manual_pinst
        self.problem = self.create_problem_and_env(manual_pinst)

        self.n_kinematic_trial_limit = 100
        self.rrt_time_limit = np.inf

    def compute_obj_collisions(self):
        robot = self.robot

        obj_names = []
        leftarm_manip = robot.GetManipulator('leftarm')
        rightarm_manip = robot.GetManipulator('rightarm')

        env = self.env
        path = self.problem['original_path']
        objs = self.objects

        assert len(robot.GetGrabbed()) == 0, "Robot must not hold anything to check obj collisions"

        if len(path) > 1000:
            path_reduced = path[0:len(path) - 1:2]  # halves the path length

        with robot:
            set_config(robot, FOLDED_LEFT_ARM, leftarm_manip.GetArmIndices())
            set_config(robot, mirror_arm_config(FOLDED_LEFT_ARM),
                       rightarm_manip.GetArmIndices())
            for p in path_reduced:
                # This would not work if the robot is holding the object
                set_robot_config(p, robot)
                for obj in objs:
                    if env.CheckCollision(robot, obj) and not (obj.GetName() in obj_names):
                        obj_names.append(obj.GetName())
                        # set_color(obj, COLLIDING_OBJ_COLOR  )
                    # else:
                    # import pdb;pdb.set_tracce()
                    # set_color(obj,OBJECT_ORIGINAL_COLOR)

        return obj_names

    def get_p_inst(self):
        if socket.gethostname() == 'dell-XPS-15-9560':
            self.problem_dir = '../AdvActorCriticNAMOResults/problems/'
            n_probs_threshold = 900
        else:
            self.problem_dir = '/data/public/rw/pass.port//NAMO/problems/'
            n_probs_threshold = 8500

        if self.manual_pinst is None:
            probs_created = [f for f in os.listdir(self.problem_dir) if f.find('.pkl') != -1]
            n_probs_created = len(probs_created)
            is_need_more_problems = n_probs_threshold > n_probs_created
            if not is_need_more_problems:
                print "Using problem ", self.problem_dir + probs_created[random.randint(1, n_probs_created - 1)]
                return self.problem_dir + probs_created[np.random.randint(n_probs_threshold - 1)]
            else:
                return self.problem_dir + 'prob_inst_' + str(n_probs_created) + '.pkl'
        else:
            return self.problem_dir + self.manual_pinst

    def create_problem_and_env(self, p_inst):
        self.env = Environment()
        self.problem = NAMO_problem(self.env, p_inst)

        while self.problem is None:
            self.env = Environment()
            self.problem = NAMO_problem(self.env, p_inst)

        self.objects = self.problem['objects']
        for obj in self.objects:
            set_color(obj, OBJECT_ORIGINAL_COLOR)

        self.target_obj = self.problem['target_obj']
        set_color(self.target_obj, TARGET_OBJ_COLOR)

        self.robot = self.env.GetRobots()[0]
        self.robot_region = self.problem['all_region']
        self.init_base_conf = np.array([-1, 1, 0])
        self.init_saver = DynamicEnvironmentStateSaver(self.env)

        # self.collided_objs = self.problem['collided_objs']
        self.collided_objs = self.compute_obj_collisions()
        for obj_name in self.collided_objs:
            set_color(self.env.GetKinBody(obj_name), COLLIDING_OBJ_COLOR)
        self.curr_obj_name = self.collided_objs[0]

    def reset_to_init_state(self):
        # stop simulation
        self.env.StopSimulation()
        self.robot.ReleaseAllGrabbed()
        self.env.StartSimulation(0.01)
        self.init_saver.Restore()
        self.collided_objs = self.compute_obj_collisions()
        self.curr_obj_name = self.collided_objs[0]

    def get_cvec(self, key_configs):
        # our state is represented with a key configuration collision vector
        c_data = compute_occ_vec(key_configs, self.robot, self.env)[None, :] * 1
        scaled_c = convert_collision_vec_to_one_hot(c_data)
        c_data = np.tile(scaled_c, (1, 1, 1))
        return c_data

    def apply_pick_action(self, action, unif=False):
        robot = self.robot
        env = self.env
        obj = self.env.GetKinBody(self.curr_obj_name)
        leftarm_manip = robot.GetManipulator('leftarm')
        rightarm_torso_manip = robot.GetManipulator('rightarm_torso')

        # Perhaps I should move  this to the predict function of pick
        grasp_params = action[0, 0:3]
        rel_pick_base_pose = action[0, 3:]
        curr_obj_xy = get_body_xytheta(obj)[0, 0:2]
        if not unif:
            abs_pick_base_pose = convert_rel_to_abs_base_pose(rel_pick_base_pose, curr_obj_xy)
            clean_pose_data(abs_pick_base_pose)
        else:
            abs_pick_base_pose = action[0, 3:]

        set_robot_config(abs_pick_base_pose, robot)
        grasps = compute_two_arm_grasp(depth_portion=grasp_params[2],
                                       height_portion=grasp_params[1],
                                       theta=grasp_params[0],
                                       obj=obj,
                                       robot=robot)
        g_config = solveTwoArmIKs(env, robot, obj, grasps)
        if g_config is None:
            return False
        pick_obj(obj, robot, g_config, leftarm_manip, rightarm_torso_manip)
        return True

    def get_motion_plan(self, goal):
        d_fn = base_distance_fn(self.robot, x_extents=2.51, y_extents=2.51)
        s_fn = base_sample_fn(self.robot, x_extents=2.51, y_extents=2.51)
        e_fn = base_extend_fn(self.robot)
        c_fn = collision_fn(self.env, self.robot)
        q_init = self.robot.GetActiveDOFValues()

        n_iterations = [20, 50, 100, 500, 1000]
        print "Path planning..."
        stime = time.time()
        for n_iter in n_iterations:
            path = rrt_connect(q_init, goal, d_fn, s_fn, e_fn, c_fn, iterations=n_iter)
            if path is not None:
                path = smooth_path(path, e_fn, c_fn)
                print "Path Found, took %.2f"%(time.time()-stime)
                return path, "HasSolution"

        print "Path not found, took %.2f"%(time.time()-stime)
        return None, 'NoPath'

    def exists_base_path(self, c0, goal):
        time_limit = self.rrt_time_limit
        robot = self.robot
        env = self.env

        if len(goal.shape) == 2: goal = goal.squeeze()
        before = get_robot_xytheta(robot).squeeze()
        set_robot_config(c0, robot)
        status = 'NoPath'
        print "checking path between ", c0, goal

        stime = time.time()
        for node_lim in [10000]:
            path, tpath, status = get_motion_plan(robot,
                                                  goal, env, n_node_lim=node_lim, t_lim=np.inf)
            print "RRT time so far", time.time() - stime, status
            if status == 'HasSolution': break
            if time.time() - stime > time_limit:
                print "Time out"
                break
        set_robot_config(before, robot)
        if status == "HasSolution":
            print "Path found"
            return True
        else:
            print "Failed to find path"
            return False

    def apply_place_action(self, action):
        robot = self.robot
        env = self.env
        obj = self.env.GetKinBody(self.curr_obj_name)
        leftarm_manip = robot.GetManipulator('leftarm')
        rightarm_manip = robot.GetManipulator('rightarm')

        place_robot_pose = action
        set_robot_config(place_robot_pose.squeeze(), robot)
        if check_collision_except(robot, obj, env):
            return False
        place_obj(obj, robot, FOLDED_LEFT_ARM, leftarm_manip, rightarm_manip)
        return True

    def visualize_placements(self, pi, cvec, misc):
        cvec = cvec.reshape((cvec.shape[0], cvec.shape[1], cvec.shape[2]))
        obj = self.env.GetKinBody(self.curr_obj_name)
        is_unif_smpler = pi.__str__().find('Uniform') != -1
        samples = pi.predict(cvec, misc, 100)
        draw_configs(samples, self.env, name='conf', transparency=0.5)
        if self.env.GetViewer() is None:
            self.env.SetViewer('qtcoin')
        raw_input('press a key to continue')
        remove_drawn_configs('conf', self.env)
        remove_drawn_configs('unif_conf', self.env)

    def visualize_pick_base(self, pick_pi, cvec, misc):
        # print misc
        cvec = cvec.reshape((cvec.shape[0], cvec.shape[1], cvec.shape[2]))
        obj = self.env.GetKinBody(self.curr_obj_name)
        is_unif_smpler = pick_pi.__str__().find('Uniform') != -1
        if is_unif_smpler:
            action = pick_pi.predict(obj, cvec, misc, 100)
            abs_pick_base_pose = action[:, 3:]
        else:
            action = pick_pi.predict(cvec, misc, 100)
            curr_obj_xy = get_body_xytheta(obj)[0, 0:2]
            relative_pick_base_pose = action[:, 3:]
            abs_pick_base_pose = convert_rel_to_abs_base_pose(relative_pick_base_pose, curr_obj_xy)

        draw_configs(abs_pick_base_pose, self.env, name='conf', transparency=0.5)
        if self.env.GetViewer() is None:
            self.env.SetViewer('qtcoin')
        raw_input('press a key to continue')
        remove_drawn_configs('conf', self.env)
        remove_drawn_configs('unif_conf', self.env)

    def sample_feasible_pick(self, pick_pi, fc, misc):
        robot = self.robot
        env = self.env
        obj = self.env.GetKinBody(self.curr_obj_name)
        leftarm_manip = robot.GetManipulator('leftarm')
        rightarm_torso_manip = robot.GetManipulator('rightarm_torso')

        feasible_pick_found = False
        for n_trial in range(self.n_kinematic_trial_limit):
            stime = time.time()
            is_unif_smpler = pick_pi.__str__().find('Uniform') != -1
            if is_unif_smpler:
                action = pick_pi.predict(obj, fc, misc)
                abs_pick_base_pose = action[0, 3:]
                if abs_pick_base_pose is None:
                    continue
            else:
                action = pick_pi.predict(fc, misc)
                curr_obj_xy = get_body_xytheta(obj)[0, 0:2]
                rel_pick_base_pose = action[0, 3:]
                abs_pick_base_pose = convert_rel_to_abs_base_pose(rel_pick_base_pose, curr_obj_xy)
            grasp_params = action[0, 0:3]
            before = get_robot_xytheta(robot).squeeze()
            if abs_pick_base_pose[0] is None:
                continue
            set_robot_config(abs_pick_base_pose, robot)
            if not self.is_feasible_base_pose(abs_pick_base_pose):
                set_robot_config(before, robot)
                continue

            set_robot_config(abs_pick_base_pose, robot)
            grasps = compute_two_arm_grasp(depth_portion=grasp_params[2], \
                                           height_portion=grasp_params[1], \
                                           theta=grasp_params[0], obj=obj, robot=robot)
            g_config = solveTwoArmIKs(env, robot, obj, grasps)
            set_robot_config(before, robot)
            if g_config is None:
                continue
            print "IK took", time.time() - stime

            print "Checking pick path existence"
            if self.exists_base_path(before, abs_pick_base_pose):
                feasible_pick_found = True
                set_robot_config(before, robot)
                return action, feasible_pick_found
            else:
                break

        set_robot_config(before, robot)
        return action, False

    def is_feasible_base_pose(self, base_pose):
        robot = self.robot
        env = self.env
        robot_region = self.robot_region
        obj = self.env.GetKinBody(self.curr_obj_name)

        with robot:
            set_robot_config(base_pose, robot)
            if not env.CheckCollision(robot) and \
                    (robot_region.contains(robot.ComputeAABB())):
                return True
            return False



