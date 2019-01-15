from problem_environments.conveyor_belt_env import ConveyorBelt
from problem_environments.namo_env import NAMO
from problem_environments.mover_env import Mover

from planners.high_level_planner import HighLevelPlanner


from sampling_strategies.voo import VOO, MoverVOO
from sampling_strategies.uniform import Uniform
from sampling_strategies.doo import DOO

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

    save_dir = ROOTDIR + '/test_results/' + domain + '_results/' + 'mcts_iter_' + str(mcts_iter) + '/uct_' \
               + str(uct_parameter) + '_widening_' \
               + str(widening_parameter) + '_' + sampling_strategy + '_n_feasible_checks_'+str(n_feasibility_checks)

    if sampling_strategy != 'unif':
        save_dir = save_dir + '/eps_' + str(sampling_strategy_exploration_parameter) + '/c1_' + str(c1) + '/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    return save_dir


def make_problem_env(domain_name):
    if domain_name == 'namo':
        problem_env = NAMO()
    elif domain_name == 'convbelt':
        problem_env = ConveyorBelt()
    else:
        problem_env = Mover()
    return problem_env


def get_task_plan(domain_name, problem_env):
    if domain_name == 'namo':
        task_plan = [{'region': problem_env.regions['loading_region'], 'objects': [problem_env.target_object]}]
    elif domain_name == 'convbelt':
        task_plan = [{'region': problem_env.regions['object_region'], 'objects': problem_env.objects}]
    else:
        packing_boxes = problem_env.packing_boxes
        task_plan = [{'region': problem_env.box_regions[packing_boxes[0].GetName()],
                      'objects': problem_env.shelf_objs[0:5]}]
    return task_plan


def main():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-uct', type=float, default=0.0)
    parser.add_argument('-widening_parameter', type=float, default=0.8)
    parser.add_argument('-epsilon', type=float, default=0.3)
    parser.add_argument('-sampling_strategy', type=str, default='unif')
    parser.add_argument('-problem_idx', type=int, default=0)
    parser.add_argument('-domain', type=str, default='namo')
    parser.add_argument('-planner', type=str, default='mcts')
    parser.add_argument('-v', action='store_true', default=False)
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-mcts_iter', type=int, default=50)
    parser.add_argument('-seed', type=int, default=50)
    parser.add_argument('-max_time', type=float, default=np.inf)
    parser.add_argument('-c1', type=float, default=1)
    parser.add_argument('-n_feasibility_checks', type=int, default=50)
    args = parser.parse_args()

    if args.debug:
        sd = args.seed
        print "RANDOM SEED SET", np.random.seed(sd)
        print "RANDOM SEED SET", random.seed(sd)

    save_dir = make_save_dir(args)
    stat_file_name = save_dir + str(args.problem_idx)+'.pkl'
    if os.path.isfile(stat_file_name):
        print "already done"
        return -1

    problem_env = make_problem_env(args.domain)
    task_plan = get_task_plan(args.domain, problem_env)

    if args.v:
        problem_env.env.SetViewer('qtcoin')

    hierarchical_planner = HighLevelPlanner(task_plan, problem_env, args.domain, args.debug)
    hierarchical_planner.set_mcts_parameters(args)
    search_time_to_reward, plan, optimal_score_achieved = hierarchical_planner.search()

    pickle.dump({'search_time': search_time_to_reward, 'plan': plan, 'pidx': args.problem_idx,
                 'is_optimal_score': optimal_score_achieved}, open(save_dir + '/' + str(args.problem_idx)+'.pkl', 'wb'))

    problem_env.problem_config['env'].Destroy()
    openravepy.RaveDestroy()


if __name__ == '__main__':
    main()
