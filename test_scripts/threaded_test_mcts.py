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
    voo_sampling_mode = config['voo_sampling_mode']
    use_uct = config['use_uct']
    add = config['add']
    n_switch = config['n_switch']

    command = 'python ./test_scripts/test_mcts.py -sampling_strategy ' + s + \
        ' -problem_idx ' + str(pidx) + ' -domain ' + d + ' -epsilon ' + str(e) + ' -w ' + str(w) + \
        ' -mcts_iter ' + str(mcts_iter) + ' -uct '+str(uct) + ' -n_feasibility_checks ' + str(n_feasibility_checks) + \
        ' -random_seed ' + str(seed) + ' -voo_sampling_mode ' + str(voo_sampling_mode) + ' -n_switch ' + str(n_switch)
    if pw:
        command += ' -pw '

    if use_uct:
        command += ' -use_uct'

    if add != '':
        command +=  ' -add ' + add

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
    parser.add_argument('-pidx', type=int, default=0)
    parser.add_argument('-random_seeds', nargs='+', type=int)
    parser.add_argument('-pw', action='store_true', default=False)
    parser.add_argument('-voo_sampling_mode', type=str, default='gaussian')
    parser.add_argument('-add', type=str, default='')
    parser.add_argument('-use_uct', action='store_true', default=False)
    parser.add_argument('-n_switch', nargs='+', type=int)

    args = parser.parse_args()

    sampling_strategy = args.sampling
    epsilons = args.epsilon if args.epsilon is not None else [-1.0]
    domain = args.domain
    widening_parameters = args.w if args.w is not None else [1]
    mcts_iter = args.mcts_iter
    ucts = args.uct if args.uct is not None else [0.0]
    n_feasibility_checks = args.n_feasibility_checks if args.n_feasibility_checks is not None else [50]
    seeds = args.random_seeds if args.random_seeds is not None else range(20)
    n_switches = args.n_switch if args.n_switch is not None else [35]

    pidx = args.pidx

    configs = []
    for n_switch in n_switches:
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
                                      'voo_sampling_mode': args.voo_sampling_mode,
                                      'pw': args.pw,
                                      'use_uct': args.use_uct,
                                      'add': args.add,
                                      'n_switch': n_switch}
                            configs.append(config)

    n_workers = int(20)
    print configs
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
