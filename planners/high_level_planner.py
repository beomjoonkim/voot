from planners.mcts import MCTS
from planners.constrained_mcts import ConstrainedMCTS
from planners.fetch_planner import FetchPlanner
from planners.namo_planner import NAMOPlanner
from planners.namo_domain_namo_planner import NamoDomainNamoPlanner

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
            self.fetch_planner = FetchPlanner(problem_env, self, n_iter=10000, n_optimal_iter=0, max_time=500)
        else:
            self.fetch_planner = FetchPlanner(problem_env, self, n_iter=10, n_optimal_iter=0)

        self.is_debugging = is_debugging

        if self.problem_env.name == 'namo':
            self.namo_planner = NamoDomainNamoPlanner(problem_env, self, n_iter=10000, n_optimal_iter=0, max_time=500)
        else:
            self.namo_planner = NAMOPlanner(problem_env, self, n_iter=30, n_optimal_iter=0)

        ## for debugging purpose, remove later
        self.env = self.problem_env.env
        self.robot = self.problem_env.robot

    def set_mcts_parameters(self, widening_parameter, uct_parameter, sampling_stategy):
        self.widening_parameter = widening_parameter
        self.uct_parameter = uct_parameter
        self.sampling_stategy = sampling_stategy
        self.mcts = MCTS(self.widening_parameter, self.uct_parameter, self.sampling_stategy,
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
            if len(self.problem_env.robot.GetGrabbed())  > 0:
                return self.problem_env.robot.GetGrabbed()[0] ==self.fetch_planner.fetching_object
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
            # todo what is the optimal reward?
            return True

    def stitch_fetch_and_namo_plans(self, fetch_plan, namo_plan):
        if len(namo_plan) == 0:
            stitched_plan = fetch_plan
        else:
            pick_path_to_target_from_last_obstacle_clearance = namo_plan[-1]['path']['pick_motion']
            fetch_plan[0]['path'] = pick_path_to_target_from_last_obstacle_clearance
            stitched_plan = namo_plan + fetch_plan
        return stitched_plan

    def todo_solve_abstract_pick_and_place_for_region(self, objects, target_packing_region):
        plan = []
        next_init_node = None
        for object in objects:
            #fetch_plan, goal_node = self.fetch_planner.solve_fetching_single_obj(object, target_packing_region, self.mcts, next_init_node)
            #pklsave(fetch_plan)
            fetch_plan = pklload()
            initial_collision_names = self.fetch_planner.get_initial_collisions(fetch_plan)

            fetch_pick_path_entrance = self.fetch_planner.get_fetch_pick_entrance_config(fetch_plan, object)
            fetch_pick_path_exit = self.fetch_planner.get_fetch_pick_exit_config(fetch_plan, object)
            fetch_place_path_entrance = self.fetch_planner.get_fetch_place_entrance_config(fetch_plan, object,
                                                                                           target_packing_region)
            self.problem_env.high_level_planner = self
            self.problem_env.reset_to_init_state(self.mcts.s0_node)
            goal_node = self.mcts.s0_node
            import pdb;pdb.set_trace()

            # define the four different NAMO problems
            self.namo_planner.initialize_namo_problem(fetch_plan, object, goal_node, fetch_pick_path_exit,
                                                      fetch_place_path_entrance, target_packing_region.name)
            # First NAMO problem up until the pick path entrance
            curr_robot_conf = fetch_plan[0]['path'][0]
            namo_plan, goal_node = self.namo_planner.solve_single_object(curr_robot_conf, fetch_pick_path_entrance,
                                                                         self.problem_env.get_region_containing(self.robot),
                                                                         initial_collision_names,
                                                                         self.mcts)
            namo_plan = pklsave(namo_plan, 'namo_plan')
            import pdb;pdb.set_trace()

            # Second NAMO problem up from pick path entrance to pick path exit
            namo_plan = pklload('namo_plan')
            self.problem_env.apply_plan(namo_plan)
            fetching_region = self.problem_env.get_region_containing(object)
            c_init = fetch_pick_path_entrance
            c_goal = fetch_pick_path_exit
            self.namo_planner.fetch_place_path = namo_plan[-1]['path']['fetching_place_motion']
            self.problem_env.set_arm_base_config(c_init)
            self.mcts.s0_node.state_saver = DynamicEnvironmentStateSaver(self.env)
            namo_plan2, goal_node2 = self.namo_planner.solve_single_object(c_init, c_goal,
                                                                           fetching_region,
                                                                           initial_collision_names, self.mcts)
            import pdb;pdb.set_trace()


            ## post-contact
            # Third NAMO problem from pick path exit to place path entrance

            # Fourth NAMO problem from place path entrance to fetch_place_conf


            print 'Solved fetching', object
            import pdb;pdb.set_trace()
            #if len(initial_collision_names) == 0:
            #    object.Enable(False)
            #    next_init_node = goal_node
            #    continue
            #goal_node.state_saver = DynamicEnvironmentStateSaver(self.problem_env.env)

            goal_node = self.mcts.s0_node
            self.namo_planner.initialize_namo_problem(fetch_plan, object, initial_collision_names, fetch_place_path,
                                                      goal_node, target_packing_region.name)
            namo_plan, goal_node = self.namo_planner.solve_single_object(object, fetch_pick_conf, fetch_place_path,
                                                                         self.mcts)
            object.Enable(False)
            # todo:
            #   restore the goal node from namo
            #   apply fetch plan
            self.mcts.switch_init_node(goal_node)
            print "Solved"

            # todo:
            #   - The plan, in case when NAMO is involved, should be arranged as follows:
            #   1. NAMO plan
            #   2. A pick operator instance, gotten from fetch_plan, but whose path field is replaced by the NAMO plan's
            #      last place step, ['path']['pick_plan'] field

            concrete_pick_plan = self.stitch_fetch_and_namo_plans(fetch_plan, namo_plan)
            import pdb;pdb.set_trace()

    def solve_convbelt(self, objects, target_packing_region):
        plan = []
        next_init_node = None
        search_time_to_reward, fetch_plan, goal_node = self.fetch_planner.solve_packing(objects,
                                                                                              target_packing_region,
                                                                                              self.mcts,
                                                                                              next_init_node)
        return search_time_to_reward, fetch_plan, goal_node

    def solve_namo(self, object, target_packing_region):
        next_init_node = None
        fetch_search_time_to_reward, fetch_plan, goal_node = self.fetch_planner.solve_fetching_single_object(object,
                                                                                                      target_packing_region,
                                                                                                      self.mcts,
                                                                                                      next_init_node)

        initial_collision_names = self.fetch_planner.get_initial_collisions(fetch_plan)

        print "Solved fetching"
        self.namo_planner.namo_domain_initialize_namo_problem(fetch_plan, goal_node)
        namo_search_time_to_reward, namo_plan, goal_node = self.namo_planner.namo_domain_solve_single_object(initial_collision_names,
                                                                                 self.mcts)
        search_time_to_reward = {'fetch': fetch_search_time_to_reward, 'namo': namo_search_time_to_reward}
        plan = fetch_plan+namo_plan
        return search_time_to_reward, plan, goal_node

    def search(self):
        for plan_step in self.abstract_task_plan:
            # get the first region
            target_packing_region = plan_step['region']
            target_packing_region.draw(self.problem_env.env)
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






