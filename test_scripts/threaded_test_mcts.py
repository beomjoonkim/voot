import os
import sys
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
import argparse
import time


def worker_p(config):
    s = config['sampling_strategy']
    d = config['domain']
    pidx = config['pidx']
    w = config['widening_parameter']
    e = config['epsilon']
    mcts_iter = config['mcts_iter']
    uct = config['uct']
    n_feasibility_checks = config['n_feasibility_checks']
    seed = config['seed']
    pw = config['pw']

    command = 'python ./test_scripts/test_mcts.py -sampling_strategy ' + s + \
        ' -problem_idx ' + str(pidx) + ' -domain ' + d + ' -epsilon ' + str(e) + ' -widening_parameter ' + str(w) + \
        ' -mcts_iter ' + str(mcts_iter) + ' -uct '+str(uct) + ' -n_feasibility_checks ' + str(n_feasibility_checks) + \
        ' -random_seed ' + str(seed)
    if pw:
        command += ' -pw '

    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    time.sleep(1)
    return worker_p(multi_args)


def main():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-sampling', type=str, default='unif')
    parser.add_argument('-domain', type=str, default='minimum_displacement_removal')
    parser.add_argument('-mcts_iter', type=int, default=1000)
    parser.add_argument('-w', nargs='+', type=float)
    parser.add_argument('-uct', nargs='+', type=float)
    parser.add_argument('-n_feasibility_checks', nargs='+', type=int)
    parser.add_argument('-epsilon', nargs='+', type=float)
    parser.add_argument('-pidxs', nargs='+', type=int)
    parser.add_argument('-random_seeds', nargs='+', type=int)
    parser.add_argument('-pw', action='store_true', default=False)

    args = parser.parse_args()

    sampling_strategy = args.sampling
    epsilons = args.epsilon if args.epsilon is not None else [-1.0]
    domain = args.domain
    widening_parameters = args.w if args.w is not None else [1]
    mcts_iter = args.mcts_iter
    ucts = args.uct if args.uct is not None else [1.0]
    n_feasibility_checks = args.n_feasibility_checks if args.n_feasibility_checks is not None else [50]

    if args.domain == 'minimum_displacement_removal':
        pidx = 0
    else:
        pass
    seeds = args.random_seeds
    configs = []
    for n_feasibility_check in n_feasibility_checks:
        for uct in ucts:
            for e in epsilons:
                for seed in seeds:
                    for widening_parameter in widening_parameters:
                        config = {"widening_parameter": widening_parameter,
                                  "epsilon": e,
                                  'pidx': pidx,
                                  'domain': domain,
                                  'sampling_strategy': sampling_strategy,
                                  'mcts_iter': mcts_iter,
                                  'uct': uct,
                                  'n_feasibility_checks': n_feasibility_check,
                                  'seed': seed,
                                  'pw': args.pw}
                        configs.append(config)

    n_workers = int(20)
    print configs
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
