import os
import sys
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
import argparse
import time


def worker_p(config):
    s = config['sampling_strategy']
    d = config['domain']
    pidx = config['trial']
    w = config['widening_parameter']
    e = config['epsilon']
    mcts_iter = config['mcts_iter']
    c1 = config['c1']
    n_feasibility_checks = config['n_feasibility_checks']
    seed = config['seed']

    command = 'python ./test_scripts/test_mcts.py -sampling_strategy ' + s + \
        ' -problem_idx ' + str(pidx) + ' -domain ' + d + ' -epsilon ' + str(e) + ' -widening_parameter ' + str(w) + \
        ' -mcts_iter ' + str(mcts_iter) + ' -c1 '+str(c1) + ' -n_feasibility_checks ' + str(n_feasibility_checks) + \
        ' -random_seed ' + str(seed)

    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    time.sleep(1)
    return worker_p(multi_args)


def main():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-sampling', type=str, default='unif')
    parser.add_argument('-domain', type=str, default='convbelt')
    parser.add_argument('-mcts_iter', type=int, default=500)
    parser.add_argument('-w', nargs='+', type=float)
    parser.add_argument('-c1', nargs='+', type=float)
    parser.add_argument('-n_feasibility_checks', nargs='+', type=int)
    parser.add_argument('-epsilon', nargs='+', type=float)
    parser.add_argument('-pidxs', nargs='+', type=int)
    parser.add_argument('-random_seeds', nargs='+', type=int)
    parser.add_argument('--pidxs_specified', action='store_true')

    args = parser.parse_args()

    sampling_strategy = args.sampling
    epsilons = args.epsilon if args.epsilon is not None else [-1.0]
    domain = args.domain
    widening_parameters = args.w if args.w is not None else [0.8]
    mcts_iter = args.mcts_iter
    c1s = args.c1 if args.c1 is not None else [1.0]
    n_feasibility_checks = args.n_feasibility_checks if args.n_feasibility_checks is not None else [50]

    if not args.pidxs_specified:
        trials = range(args.pidxs[0], args.pidxs[1])
    else:
        trials = args.pidxs

    seeds = range(args.random_seeds[0], args.random_seeds[1])
    configs = []
    for n_feasibility_check in n_feasibility_checks:
        for c1 in c1s:
            for e in epsilons:
                for t in trials:
                    for seed in seeds:
                        for widening_parameter in widening_parameters:
                            config = {"widening_parameter": widening_parameter,
                                      "epsilon": e, 'trial': t,
                                      'domain': domain,
                                      'sampling_strategy': sampling_strategy,
                                      'mcts_iter': mcts_iter,
                                      'c1': c1,
                                      'n_feasibility_checks': n_feasibility_check,
                                      'seed': seed}
                            configs.append(config)

    n_workers = int(20)
    print configs
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
