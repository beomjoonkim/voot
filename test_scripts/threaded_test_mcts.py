import os
import sys
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading


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
    domain = sys.argv[1]
    sampling_strategy = sys.argv[2]
    if sampling_strategy == 'voo':
        epsilons = [1.0]
        trials = range(30)
        configs = []
        for e in epsilons:
            for t in trials:
                configs.append([e, t, domain, sampling_strategy])
    else:
        trials = range(100)
        configs = []
        for t in trials:
            configs.append([t, domain, sampling_strategy])

    n_workers = int(30)

    print configs
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
