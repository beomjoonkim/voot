from planners.mcts import MCTS
from planners.fetch_planner import FetchPlanner
from planners.namo_planner import NAMOPlanner
from planners.namo_domain_namo_planner import NamoDomainNamoPlanner


import sys
from mover_library.utils import *

import pickle
import numpy as np


def pklsave(obj, name=''):
    pickle.dump(obj, open('tmp'+str(name)+'.pkl', 'wb'))


def pklload(name=''):
    return pickle.load(open('tmp'+str(name)+'.pkl', 'r'))


class HighLevelPlanner:
    def __init__(self, task_plan, problem_env, domain_name, is_debugging=False):
        self.abstract_task_plan = task_plan
        self.problem_env = problem_env
        self.domain_name = domain_name
        self.widening_parameter = None
        self.uct_parameter = None
        self.sampling_stategy = None

        self.task_plan_idx = 0
        self.obj_plan_idx = 0
        self.task_plan = task_plan

        """
        if self.problem_env.name == 'convbelt':
            self.fetch_planner = FetchPlanner(problem_env, self)
        else:
            self.fetch_planner = FetchPlanner(problem_env, self)

        self.is_debugging = is_debugging

        if self.problem_env.name == 'minimum_displacement_removal':
            #todo why is this needed?
            self.namo_planner = NamoDomainNamoPlanner(problem_env, self)
        else:
            self.namo_planner = NAMOPlanner(problem_env, self)
        """

        ## for debugging purpose, remove later
        self.env = self.problem_env.env
        self.robot = self.problem_env.robot

    def save_results(self, search_time_rwd, plan, rewards_list, iteration):
        pickle.dump({'search_time': search_time_rwd, 'plan': plan, 'iteration': iteration,
                     'reward_list': rewards_list,
                     }, open(self.stat_file_name, 'wb'))

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
        if self.problem_env.is_solving_packing:
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

    def set_object_index(self, object_index):
        self.obj_plan_idx = object_index

    def reset_task_plan_indices(self):
        self.obj_plan_idx = 0
        self.task_plan_idx = 0

    def is_goal_reached(self):
        if self.problem_env.is_solving_namo:
            return len(self.task_plan[0]['objects']) == 0 # done if all obstacles are cleared
        else:
            return self.obj_plan_idx >= len(self.problem_env.objects)

    def get_next_obj(self):
        return self.task_plan[self.task_plan_idx]['objects'][self.obj_plan_idx]

    def get_next_region(self):
        return self.task_plan[self.task_plan_idx]['region']

    def is_optimal_score_achieved(self, best_traj_rwd):
        n_objs_to_manipulate = np.sum([len(p['objects']) for p in self.task_plan])

        if self.problem_env.is_solving_fetching:
            return best_traj_rwd == 2
        elif self.problem_env.is_solving_packing:
            return self.obj_plan_idx >= len(self.problem_env.objects)
        elif self.problem_env.is_solving_namo:
            return False

    def solve_convbelt(self, objects, target_packing_region):
        plan = []
        next_init_node = None
        """
        while self.problem_env.infeasible_reward == rwd:
            pick_pi = PickWithBaseUnif(self.problem_env)
            pick_action = pick_pi.predict(objects[0], self.problem_env.regions['entire_region'], n_iter=10000)
            _, rwd, _, _ = self.problem_env.apply_two_arm_pick_action(pick_action, self.mcts.s0_node, True, None)


        checker = PickFeasibilityChecker(self.problem_env)
        checked = False
        while not checked:
            domain = get_pick_domain
            dim_parameters = domain.shape[-1]
            domain_min = domain[0]
            domain_max = domain[1]
            np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()
            pick_action = checker.check_feasibility()
            _, rwd, _, _ = self.problem_env.apply_two_arm_pick_action(pick_action, self.mcts.s0_node, True, None)
        """
        #self.mcts.s0_node = self.mcts.create_node(None, depth=0, reward=0, objs_in_collision=None, is_init_node=True)
        self.mcts.tree.root = self.mcts.s0_node
        self.problem_env.is_solving_packing = True
        search_time_to_reward, fetch_plan, goal_node, reward_list = self.fetch_planner.solve_packing(objects,
                                                                                        target_packing_region,
                                                                                        self.mcts,
                                                                                        next_init_node)
        return search_time_to_reward, fetch_plan, goal_node, reward_list

    def solve_minimum_displacement_removal(self):

        """
        self.problem_env.disable_objects()
        fetching_path,status = self.problem_env.get_base_motion_plan(self.problem_env.problem_config['goal_base_config'], region_name='entire_region')
        self.problem_env.enable_objects()
        """
        swept_volume_file_name = './problem_environments/mover_domain_problems/fetching_path_' + \
                                      str(self.problem_env.problem_idx) +'.pkl'
        swept_volume_to_clear_obstacles_from = pickle.load(open(swept_volume_file_name, 'r'))
        initial_collisions = self.problem_env.get_objs_in_collision(swept_volume_to_clear_obstacles_from,
                                                                    'entire_region')
        initial_collision_names = [o.GetName() for o in initial_collisions]
        print len(initial_collision_names)
        print "Solved fetching"
        import pdb;pdb.set_trace()
        self.namo_planner.fetch_pick_path = fetching_path
        namo_search_time_to_reward, namo_plan, goal_node, reward_list = self.namo_planner.namo_domain_solve_single_object(
                                                                                 initial_collision_names,
                                                                                 self.mcts)
        search_time_to_reward = namo_search_time_to_reward
        if namo_plan is None:
            plan = None
        else:
            plan = namo_plan
        return search_time_to_reward, plan, goal_node, reward_list

    def search(self):
        import pdb;pdb.set_trace()
        # do I even need this abstract task plan stuff for minumum constraint removal problem?


        for plan_step in self.abstract_task_plan:
            # get the first region
            target_packing_region = plan_step['region']
            #target_packing_region.draw(self.problem_env.env)
            # get the list of objects to be packed
            objects = plan_step['objects']

            # get the object ordering
            obj_plan = self.compute_object_ordering(objects)
            if self.problem_env.name == 'convbelt':
                search_time_to_reward, fetch_plan, _, reward_list = self.solve_convbelt(objects, target_packing_region)
            elif self.problem_env.name == 'minimum_displacement_removal':
                search_time_to_reward, fetch_plan,\
                    _, reward_list = self.solve_minimum_displacement_removal()
            else:
                raise NotImplementedError
        return search_time_to_reward, fetch_plan, True, reward_list

    @staticmethod
    def compute_object_ordering(objects):
        return objects






