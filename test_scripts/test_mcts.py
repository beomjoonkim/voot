from problem_environments.conveyor_belt_env import ConveyorBelt
from problem_environments.minimum_displacement_removal import MinimumDisplacementRemoval
from problem_environments.mover_env import Mover

from problem_instantiators.minimum_constraint_removal_instantiator import MinimumConstraintRemovalInstantiator
from planners.mcts import MCTS

import argparse
import cPickle as pickle
import os
import numpy as np
import random
import socket
import openravepy

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
        save_dir = ROOTDIR + '/test_results/' + domain + '_results/' + 'mcts_iter_'\
                   + str(mcts_iter) + '/uct_' \
                   + str(uct_parameter) + '_widening_' \
                   + str(widening_parameter) + '_' + sampling_strategy + '_n_feasible_checks_'+str(n_feasibility_checks) + '/'
    elif domain == 'convbelt':
        save_dir = ROOTDIR + '/test_results//' + domain + '_results//' + 'mcts_iter_' + str(mcts_iter) + '/uct_' \
                   + str(uct_parameter) + '_widening_' \
                   + str(widening_parameter) + '_' + sampling_strategy + '_n_feasible_checks_'+str(n_feasibility_checks) + '/'
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


def instantiate_mcts(args, problem_env):
    uct_parameter = args.uct
    widening_parameter = args.widening_parameter
    sampling_strategy = args.sampling_strategy
    sampling_strategy_exploration_parameter = args.epsilon
    n_feasibility_checks = args.n_feasibility_checks
    c1 = args.c1
    domain_name = args.domain

    mcts = MCTS(widening_parameter, uct_parameter, sampling_strategy,
                sampling_strategy_exploration_parameter, c1, n_feasibility_checks,
                problem_env, domain_name)
    return mcts


def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)


def main():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-uct', type=float, default=1.0)
    parser.add_argument('-widening_parameter', type=float, default=2.0)
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
    parser.add_argument('-c1', type=float, default=1) # weight for measuring distances in SE(2)
    parser.add_argument('-n_feasibility_checks', type=int, default=50)
    parser.add_argument('-random_seed', type=int, default=-1)

    args = parser.parse_args()
    if args.random_seed == -1:
        args.random_seed = args.problem_idx

    print "Problem number ", args.problem_idx
    print "Random seed set: ", args.random_seed
    set_random_seed(args.random_seed)

    save_dir = make_save_dir(args)
    stat_file_name = save_dir + '/rand_seed_' + str(args.random_seed) + '_pidx_' + str(args.problem_idx)+'.pkl'
    if os.path.isfile(stat_file_name):
        print "already done"
        return -1

    if args.domain == 'minimum_displacement_removal':
        problem_instantiator = MinimumConstraintRemovalInstantiator(args.domain)
    else:
        raise NotImplementedError
    if args.v:
        problem_instantiator.environment.env.SetViewer('qtcoin')

    mcts = instantiate_mcts(args, problem_instantiator.environment)
    search_time_to_reward, plan = mcts.search(args.mcts_iter)

    pickle.dump({'search_time': search_time_to_reward, 'plan': plan, 'pidx': args.problem_idx},
                open(stat_file_name, 'wb'))

    problem_instantiator.environment.env.Destroy()
    openravepy.RaveDestroy()


if __name__ == '__main__':
    main()
