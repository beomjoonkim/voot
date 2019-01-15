import os
import sys
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
import argparse


def worker_p(config):
    n_iter = config[0]
    problem_idx = config[1]
    algo_name = config[2]
    dim = config[3]
    command = 'python ./test_scripts/test_optimization_algorithms.py ' + str(problem_idx) + ' ' +algo_name \
              + ' ' + str(dim) + ' ' + str(n_iter)

    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    algo_name = sys.argv[1]
    dims = [int(sys.argv[2])]
    n_iter = sys.argv[3]
    pidxs = sys.argv[4].split(',')
    pidxs = range(int(pidxs[0]),int(pidxs[1]))

    configs= []
    for dim in dims:
        for t in pidxs:
            configs.append([n_iter, t, algo_name, dim])
    if algo_name == 'gpucb':
        n_workers = int(10)
    else:
        n_workers = int(30)

    print configs
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
