import os
import sys
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
import argparse


def worker_p(config):
    sampling_strategy = config[-1]
    domain = config[-2]
    pidx = config[-3]
    if sampling_strategy == 'voo':
        epsilon = config[0]
        command = 'python ./test_scripts/test_hierarchical_mcts.py -sampling_strategy ' + sampling_strategy + \
            ' -problem_idx ' + str(pidx) + ' -domain ' + domain + ' -epsilon ' + str(epsilon)
    else:
        command = 'python ./test_scripts/test_hierarchical_mcts.py -sampling_strategy ' + sampling_strategy + \
                  ' -problem_idx ' + str(pidx) + ' -domain ' + domain
    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-sampling', type=str, default='unif')
    parser.add_argument('-domain', type=str, default='namo')
    parser.add_argument('-widening_parameter', nargs='+')
    parser.add_argument('-epsilon', nargs='+')


    args = parser.parse_args()

    sampling_strategy = args.sampling_strategy
    epsilon = args.epsilon
    domain = args.domain
    widening_parameters = args.widening_parameter
    import pdb;pdb.set_trace()

    if sampling_strategy == 'voo':
        epsilons = [0.3]
        trials = range(120)
        configs = []
        for e in epsilons:
            for t in trials:
                config = {"widening_parameter": widening_parameter,
                          "epsilon": e, 'trial':t, 'domain':domain, 'sampling':sampling_strategy}
                configs.append([e, t, domain, sampling_strategy])
    else:
        trials = range(120)
        configs = []
        for t in trials:
            config = {"widening_parameter": widening_parameter,
                      "epsilon": None, 'trial':t, 'domain':domain, 'sampling':sampling_strategy}
            configs.append([t, domain, sampling_strategy])

    n_workers = int(30)

    print configs
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
