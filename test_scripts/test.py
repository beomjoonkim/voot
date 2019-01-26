from manipulation.bodies.bodies import box_body
from problem_environments.conveyor_belt_problem import create_conveyor_belt_problem
from operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp
from problem_environments.conveyor_belt_env import ConveyorBelt

from mover_library.utils import get_pick_domain, get_pick_base_pose_and_grasp_from_pick_parameters, set_robot_config, \
     two_arm_pick_object, visualize_path, set_obj_xytheta, get_body_xytheta
from mover_library.samplers import randomly_place_in_region
import numpy as np

problem = ConveyorBelt(problem_idx=1)
env=problem.env
robot = env.GetRobots()[0]
env.SetViewer('qtcoin')


tobj = env.GetKinBody('tobj3')
tobj_xytheta = get_body_xytheta(tobj.GetLinks()[1])
tobj_xytheta[0,-1] = (160/180.0)*np.pi
set_obj_xytheta(tobj_xytheta, tobj.GetLinks()[1])
for tobj in env.GetBodies():
    if tobj.GetName().find('tobj') ==-1:continue
    randomly_place_in_region(env, tobj, problem.problem_config['conveyor_belt_region'])
    pick_domain = get_pick_domain()

tobj = env.GetKinBody('tobj4')


def sample_grasp(target_obj):
    g_config = None
    i= 0
    while g_config is None:
        print i
        pick_parameter = np.random.uniform(pick_domain[0], pick_domain[1], 6)
        grasp_params, pick_base_pose = get_pick_base_pose_and_grasp_from_pick_parameters(target_obj, pick_parameter)
        set_robot_config(pick_base_pose, robot)
        i+=1
        if env.CheckCollision(robot):
            continue
        grasps = compute_two_arm_grasp(grasp_params[2], grasp_params[1], grasp_params[0], target_obj, robot)
        g_config = solveTwoArmIKs(env, robot, target_obj, grasps)
        print g_config, grasp_params
    return g_config, pick_base_pose, grasp_params

target = env.GetKinBody('matrice')
target=tobj
g_config, pick_base_pose, grasp_params = sample_grasp(target)
import pdb; pdb.set_trace()

pick_action = {'operator_name': 'two_arm_pick', 'base_pose': pick_base_pose, 'grasp_params': grasp_params, 'g_config': g_config}
two_arm_pick_object(target, robot, pick_action)
init_base_conf = np.array([0, 1.05, 0])
set_robot_config(init_base_conf, robot)
goal_config = np.array([-2,-2,np.pi/2])
place_path, status = problem.get_base_motion_plan(goal_config)

