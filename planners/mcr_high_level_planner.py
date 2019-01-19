from mover_library.utils import se2_distance, get_body_xytheta

class MCRHighLevelPlanner:
    def __init__(self, problem_env, domain_name, is_debugging):
        self.problem_env = problem_env
        self.domain_name = domain_name
        self.is_debugging = is_debugging
        self.robot = self.problem_env.robot

    def update_task_plan_indices(self, reward, operator_used):
        pass

    def is_optimal_score_achieved(self, best_traj_rwd):
        return False
        pass

    def get_next_obj(self):
        pass

    def get_next_region(self):
        pass


    def is_goal_reached(self):
        curr_xytheta = get_body_xytheta(self.robot)
        distance = se2_distance(curr_xytheta, self.problem_env.goal_base_conf, 1, 1)

        if distance < 1:
            return True
        else:
            return False






