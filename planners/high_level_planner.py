from planners.mcts import MCTS
from planners.constrained_mcts import ConstrainedMCTS
from planners.fetch_planner import FetchPlanner
from planners.namo_planner import NAMOPlanner
from planners.namo_domain_namo_planner import NamoDomainNamoPlanner
from generators.PickUniform import PickWithBaseUnif

from manipulation.primitives.savers import DynamicEnvironmentStateSaver

import sys
sys.path.append('../mover_library/')
from utils import *

import pickle
import numpy as np
import time


def pklsave(obj, name=''):
    pickle.dump(obj, open('tmp'+str(name)+'.pkl', 'wb'))


def pklload(name=''):
    return pickle.load(open('tmp'+str(name)+'.pkl', 'r'))


class HighLevelPlanner:
    def __init__(self, task_plan, problem_env, domain_name, is_debugging):
        self.abstract_task_plan = task_plan
        self.problem_env = problem_env
        self.domain_name = domain_name
        self.widening_parameter = None
        self.uct_parameter = None
        self.sampling_stategy = None
        self.task_plan_idx = 0
        self.obj_plan_idx = 0
        self.task_plan = task_plan
        if self.problem_env.name == 'convbelt':
            self.fetch_planner = FetchPlanner(problem_env, self)
        else:
            self.fetch_planner = FetchPlanner(problem_env, self)

        self.is_debugging = is_debugging

        if self.problem_env.name == 'namo':
            self.namo_planner = NamoDomainNamoPlanner(problem_env, self)
        else:
            self.namo_planner = NAMOPlanner(problem_env, self)

        ## for debugging purpose, remove later
        self.env = self.problem_env.env
        self.robot = self.problem_env.robot

    def set_mcts_parameters(self, args):
        domain = args.domain
        uct_parameter = args.uct
        widening_parameter = args.widening_parameter
        sampling_strategy = args.sampling_strategy
        sampling_strategy_exploration_parameter = args.epsilon
        mcts_iter = args.mcts_iter
        n_feasibility_checks = args.n_feasibility_checks
        c1 = args.c1

        self.widening_parameter = widening_parameter
        self.uct_parameter = uct_parameter
        self.sampling_stategy = sampling_strategy
        self.n_iter = mcts_iter
        self.n_optimal_iter = 0
        self.max_time = args.max_time
        self.mcts = MCTS(self.widening_parameter, self.uct_parameter, self.sampling_stategy,
                         sampling_strategy_exploration_parameter, c1, n_feasibility_checks,
                         self.problem_env, self.domain_name, self)

    def update_task_plan_indices(self, reward, operator_used):
        if self.problem_env.is_solving_fetching:
            made_progress = reward > 0 and operator_used.find('place') != -1
            if made_progress:
                self.obj_plan_idx += 1
                if self.obj_plan_idx == len(self.task_plan[self.task_plan_idx]['objects']):
                    self.obj_plan_idx = 0
                    self.task_plan_idx += 1
        elif self.problem_env.is_solving_namo:
            pass

    def set_task_plan(self, task_plan):
        self.task_plan = task_plan
        self.reset_task_plan_indices()

    def reset_task_plan_indices(self):
        self.obj_plan_idx = 0
        self.task_plan_idx = 0

    def is_goal_reached(self):
        if self.problem_env.name == 'namo' and self.problem_env.is_solving_fetching:
            if len(self.problem_env.robot.GetGrabbed()) > 0:
                return self.problem_env.robot.GetGrabbed()[0] == self.fetch_planner.fetching_object
            else:
                return False
        if self.problem_env.is_solving_namo:
            return len(self.task_plan[0]['objects']) == 0 # done if all obstacles are cleared
        else:
            return self.task_plan_idx >= len(self.task_plan)

    def get_next_obj(self):
        try:
            return self.task_plan[self.task_plan_idx]['objects'][self.obj_plan_idx]
        except:
            import pdb;pdb.set_trace()

    def get_next_region(self):
        return self.task_plan[self.task_plan_idx]['region']

    def is_optimal_score_achieved(self, best_traj_rwd):
        n_objs_to_manipulate = np.sum([len(p['objects']) for p in self.task_plan])

        if self.problem_env.is_solving_fetching:
            return best_traj_rwd == 2
        elif self.problem_env.is_solving_packing:
            return best_traj_rwd == n_objs_to_manipulate
        elif self.problem_env.is_solving_namo:
            return False

    def solve_convbelt(self, objects, target_packing_region):
        plan = []
        next_init_node = None
        rwd = self.problem_env.infeasible_reward
        while self.problem_env.infeasible_reward == rwd:
            pick_pi = PickWithBaseUnif(self.problem_env)
            pick_action = pick_pi.predict(objects[0], self.problem_env.regions['entire_region'], n_iter=10000)
            _, rwd, _, _ = self.problem_env.apply_two_arm_pick_action(pick_action, self.mcts.s0_node, True, None)
        self.mcts.s0_node = self.mcts.create_node(None, depth=0, reward=0, objs_in_collision=None, is_init_node=True)
        self.mcts.tree.root = self.mcts.s0_node
        search_time_to_reward, fetch_plan, goal_node = self.fetch_planner.solve_packing(objects,
                                                                                        target_packing_region,
                                                                                        self.mcts,
                                                                                        next_init_node)
        return search_time_to_reward, fetch_plan, goal_node

    def solve_namo(self, object, target_packing_region):
        next_init_node = None
        #fetch_search_time_to_reward, fetch_plan, goal_node = self.fetch_planner.solve_fetching_single_object(object,
        #                                                                                              target_packing_region,
        #                                                                                              self.mcts,
        #                                                                                              next_init_node)

        self.problem_env.disable_objects()
        fetching_path,_ = self.problem_env.get_base_motion_plan(self.problem_env.goal_base_conf,'entire_region')

        """
        object[0].Enable(True)
        pick_pi = PickWithBaseUnif(self.problem_env)
        pick_action = pick_pi.predict(object[0], self.problem_env.regions['entire_region'], n_iter=10000)
        fetching_path, status= self.problem_env.get_base_motion_plan(pick_action['base_pose'])
        self.namo_planner.fetch_pick_path = fetching_path
        """
        self.problem_env.enable_objects()
        self.namo_planner.fetch_pick_path = fetching_path

        initial_collisions = self.problem_env.get_objs_in_collision(fetching_path, 'entire_region')
        initial_collision_names = [o.GetName() for o in initial_collisions]
        print len(initial_collision_names)

        import pdb;pdb.set_trace()
        print "Solved fetching"
        #self.namo_planner.namo_domain_initialize_namo_problem(fetch_plan, goal_node)
        namo_search_time_to_reward, namo_plan, goal_node = self.namo_planner.namo_domain_solve_single_object(
                                                                                 initial_collision_names,
                                                                                 self.mcts)
        search_time_to_reward = {'fetch': [], 'namo': namo_search_time_to_reward}
        if namo_plan is None:
            plan = None
        else:
            plan = namo_plan
        return search_time_to_reward, plan, goal_node

    def search(self):
        for plan_step in self.abstract_task_plan:
            # get the first region
            target_packing_region = plan_step['region']
            #target_packing_region.draw(self.problem_env.env)
            # get the list of objects to be packed
            objects = plan_step['objects']

            # get the object ordering
            obj_plan = self.compute_object_ordering(objects)
            if self.problem_env.name == 'convbelt':
                search_time_to_reward, fetch_plan, _ = self.solve_convbelt(objects, target_packing_region)
            elif self.problem_env.name == 'namo':
                search_time_to_reward, fetch_plan, _ = self.solve_namo(objects, target_packing_region)
        return search_time_to_reward, fetch_plan, True

    @staticmethod
    def compute_object_ordering(objects):
        return objects






