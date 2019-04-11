from mover_library.samplers import *
from mover_library.utils import set_robot_config, grab_obj, release_obj


class PlaceFeasibilityChecker:
    def __init__(self, problem_env):
        self.problem_env = problem_env
        self.env = problem_env.env
        self.robot = self.env.GetRobots()[0]
        self.robot_region = self.problem_env.regions['entire_region']

    def check_feasibility(self, node, place_parameters):
        # Note:
        #    this function checks if the target region contains the robot when we place object at place_parameters
        #    and whether the robot will be in collision
        # todo: change its name
        import pdb;pdb.set_trace()
        obj = self.robot.GetGrabbed()[0]

        obj_region = node.operator_skeleton.discrete_parameters['region']
        obj_pose = place_parameters

        T_r_wrt_o = np.dot(np.linalg.inv(obj.GetTransform()), self.robot.GetTransform())
        original_trans = self.robot.GetTransform()
        original_obj_trans = obj.GetTransform()

        if self.problem_env.is_solving_namo:
            target_obj_region = self.problem_env.get_region_containing(obj)
            target_robot_region = target_obj_region  # for namo, you want to stay in the same region
        else:
            target_robot_region = self.problem_env.regions['entire_region']
            target_obj_region = obj_region  # for fetching, you want to move it around
        robot_xytheta = self.compute_robot_base_pose_given_object_pose(obj, self.robot, obj_pose, T_r_wrt_o)
        set_robot_config(robot_xytheta, self.robot)

        is_base_pose_feasible = not (
                                self.env.CheckCollision(obj) or self.env.CheckCollision(self.robot)) and \
                                (target_robot_region.contains(self.robot.ComputeAABB())) and \
                                (target_obj_region.contains(obj.ComputeAABB()))

        self.robot.SetTransform(original_trans)
        obj.SetTransform(original_obj_trans)

        if is_base_pose_feasible:
            action = {'operator_name': 'two_arm_place', 'base_pose': robot_xytheta, 'object_pose': obj_pose,
                      'action_parameters': place_parameters}
            return action, 'HasSolution'
        else:
            action = {'operator_name': 'two_arm_place', 'base_pose': None, 'object_pose': None,
                      'action_parameters': place_parameters}
            return action, 'NoSolution'

    @staticmethod
    def compute_robot_base_pose_given_object_pose(obj, robot, obj_pose, T_r_wrt_o):
        original_robot_T = robot.GetTransform()
        release_obj(robot, obj)
        set_obj_xytheta(obj_pose, obj)
        new_T_robot = np.dot(obj.GetTransform(), T_r_wrt_o)
        robot.SetTransform(new_T_robot)
        robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
        robot_xytheta = robot.GetActiveDOFValues()
        grab_obj(robot, obj)
        robot.SetTransform(original_robot_T)
        return robot_xytheta
    """
    def predict(self, obj, obj_region, n_iter):
        original_trans = self.robot.GetTransform()
        original_config = self.robot.GetDOFValues()
        T_r_wrt_o = np.dot(np.linalg.inv(obj.GetTransform()), self.robot.GetTransform())

        # obj_region is the task-level object region - where you want it to be in the task plan
        if self.problem_env.is_solving_namo:
            target_obj_region = self.problem_env.get_region_containing(obj)
            target_robot_region = target_obj_region # for namo, you want to stay in the same region
        else:
            target_robot_region = self.robot_region
            target_obj_region = obj_region # for fetching, you want to move it around

        print "Sampling place"
        for _ in range(n_iter):
            obj_pose, robot_xytheta = self.get_placement(obj, target_obj_region, T_r_wrt_o)
            set_robot_config(robot_xytheta, self.robot)
            if not (self.env.CheckCollision(obj) or self.env.CheckCollision(self.robot)) \
                    and (target_robot_region.contains(self.robot.ComputeAABB())):
                self.robot.SetTransform(original_trans)
                self.robot.SetDOFValues(original_config)
                return {'operator_name': 'two_arm_place', 'base_pose': robot_xytheta, 'object_pose': obj_pose}
            else:
                self.robot.SetTransform(original_trans)
                self.robot.SetDOFValues(original_config)

        self.robot.SetTransform(original_trans)
        self.robot.SetDOFValues(original_config)
        print "Sampling failed"
        return {'operator_name': 'two_arm_place', 'base_pose': None, 'object_pose': None}

    def get_placement(self, obj, target_obj_region, T_r_wrt_o):
        original_trans = self.robot.GetTransform()
        original_config = self.robot.GetDOFValues()
        self.robot.SetTransform(original_trans)
        self.robot.SetDOFValues(original_config)

        release_obj(self.robot, obj)
        with self.robot:
            # print target_obj_region
            obj_pose = randomly_place_in_region(self.env, obj, target_obj_region)  # randomly place obj
            obj_pose = obj_pose.squeeze()

            # compute the resulting robot transform
            new_T_robot = np.dot(obj.GetTransform(), T_r_wrt_o)
            self.robot.SetTransform(new_T_robot)
            self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
            robot_xytheta = self.robot.GetActiveDOFValues()
            set_robot_config(robot_xytheta, self.robot)
            grab_obj(self.robot, obj)
        return obj_pose, robot_xytheta
    """

