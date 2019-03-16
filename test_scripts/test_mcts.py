from problem_environments.conveyor_belt_env import ConveyorBelt
from problem_environments.minimum_displacement_removal import MinimumDisplacementRemoval
from problem_environments.mover_env import Mover

from planners.high_level_planner import HighLevelPlanner
from planners.mcts import MCTS

import argparse
import cPickle as pickle
import os
import openravepy
import numpy as np
import random
import socket

hostname = socket.gethostname()
if hostname == 'dell-XPS-15-9560' or hostname=='phaedra':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/gtamp_results/'


def make_save_dir(args):
    domain = args.domain
    uct_parameter = args.uct
    widening_parameter = args.widening_parameter
    sampling_strategy = args.sampling_strategy
    sampling_strategy_exploration_parameter = args.epsilon
    mcts_iter = args.mcts_iter
    n_feasibility_checks = args.n_feasibility_checks
    c1 = args.c1

    if domain == 'minimum_displacement_removal':
        save_dir = ROOTDIR + '/test_results//root_switching/no_infeasible_place/no_going_back_to_s0_no_switch_counter/' + domain + '_results/' + 'mcts_iter_'\
                   + str(mcts_iter) + '/uct_' \
                   + str(uct_parameter) + '_widening_' \
                   + str(widening_parameter) + '_' + sampling_strategy + '_n_feasible_checks_'+str(n_feasibility_checks) + '/'
    elif domain == 'convbelt':
        save_dir = ROOTDIR + '/test_results//' + domain + '_results//' + 'mcts_iter_' + str(mcts_iter) + '/uct_' \
                   + str(uct_parameter) + '_widening_' \
                   + str(widening_parameter) + '_' + sampling_strategy + '_n_feasible_checks_'+str(n_feasibility_checks) + '/'

    elif domain == 'mcr':
        save_dir = ROOTDIR + '/test_results/' + domain + '_results/' + 'mcts_iter_' \
                   + str(mcts_iter) + '/uct_' \
                   + str(uct_parameter) + '_widening_' \
                   + str(widening_parameter) + '_' + sampling_strategy + \
                   '_n_feasible_checks_' + str(n_feasibility_checks) + '/'
    else:
        raise NotImplementedError

    if sampling_strategy != 'unif':
        save_dir = save_dir + '/eps_' + str(sampling_strategy_exploration_parameter) + '/c1_' + str(c1) + '/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    return save_dir


def make_problem_env(domain_name, problem_idx):
    if domain_name == 'minimum_displacement_removal':
        problem_env = MinimumDisplacementRemoval(problem_idx)
    elif domain_name == 'convbelt':
        problem_env = ConveyorBelt(problem_idx)
    else:
        problem_env = Mover()
    return problem_env


def get_task_plan(domain_name, problem_env):
    if domain_name == 'minimum_displacement_removal':
        task_plan = [{'region': problem_env.regions['entire_region'], 'objects': [problem_env.objects[0]]}] # dummy
    elif domain_name == 'convbelt':
        task_plan = [{'region': problem_env.regions['object_region'], 'objects': problem_env.objects}]
    else:
        packing_boxes = problem_env.packing_boxes
        task_plan = [{'region': problem_env.box_regions[packing_boxes[0].GetName()],
                      'objects': problem_env.shelf_objs[0:5]}]
    return task_plan


def instantiate_mcts(args, problem_env, domain_name, high_level_planner):
    uct_parameter = args.uct
    widening_parameter = args.widening_parameter
    sampling_strategy = args.sampling_strategy
    sampling_strategy_exploration_parameter = args.epsilon
    n_feasibility_checks = args.n_feasibility_checks
    c1 = args.c1

    mcts = MCTS(widening_parameter, uct_parameter, sampling_strategy,
                sampling_strategy_exploration_parameter, c1, n_feasibility_checks,
                problem_env, domain_name, high_level_planner)
    return mcts


def main():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-uct', type=float, default=0.0)
    parser.add_argument('-widening_parameter', type=float, default=0.8)
    parser.add_argument('-epsilon', type=float, default=0.3)
    parser.add_argument('-sampling_strategy', type=str, default='unif')
    parser.add_argument('-problem_idx', type=int, default=0)
    parser.add_argument('-domain', type=str, default='minimum_displacement_removal')
    parser.add_argument('-planner', type=str, default='mcts')
    parser.add_argument('-v', action='store_true', default=False)
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-mcts_iter', type=int, default=500)
    parser.add_argument('-seed', type=int, default=50)
    parser.add_argument('-max_time', type=float, default=np.inf)
    parser.add_argument('-c1', type=float, default=1)
    parser.add_argument('-n_feasibility_checks', type=int, default=50)
    parser.add_argument('-random_seed', type=int, default=-1)

    args = parser.parse_args()
    if args.random_seed == -1:
        args.random_seed = args.problem_idx

    print "Problem number ", args.problem_idx
    print "Random seed set: ", args.random_seed
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    save_dir = make_save_dir(args)
    stat_file_name = save_dir + '/rand_seed_' + str(args.random_seed) + '_pidx_' + str(args.problem_idx)+'.pkl'
    if os.path.isfile(stat_file_name):
        print "already done"
        return -1

    problem_env = make_problem_env(args.domain, args.problem_idx)
    if args.v:
        problem_env.env.SetViewer('qtcoin')
    task_plan = get_task_plan(args.domain, problem_env)
    hierarchical_planner = HighLevelPlanner(task_plan, problem_env, args.domain, args.debug)
    hierarchical_planner.set_mcts_parameters(args)
    hierarchical_planner.stat_file_name = stat_file_name

    search_time_to_reward, plan, optimal_score_achieved, reward_list = hierarchical_planner.search()
    # todo check if search_time_to_reward returns a solution

    pickle.dump({'search_time': search_time_to_reward, 'plan': plan, 'pidx': args.problem_idx,
                 'reward_list': reward_list,
                 'is_optimal_score': optimal_score_achieved}, open(stat_file_name, 'wb'))

    problem_env.problem_config['env'].Destroy()
    openravepy.RaveDestroy()


if __name__ == '__main__':
    main()
