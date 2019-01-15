import os
import sys
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
import argparse


def worker_p(config):
    problem_idx = config[0]
    algo_name = config[1]
    dim = config[2]
    command = 'python ./test_scripts/test_optimization_algorithms.py ' + str(problem_idx) + ' ' +algo_name \
              + ' ' + str(dim)

    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    algo_name = sys.argv[1]
    if algo_name == 'gpucb':
      dims = [int(sys.argv[2])]
      pidxs = sys.argv[3].split(',')
      pidxs = range(int(pidxs[0]),int(pidxs[1]))
    else:
      dims = [2,6,10,20]
      pidxs = range(200)

    configs= []
    for dim in dims:
        for t in pidxs:
            configs.append([t, algo_name, dim])
    if algo_name == 'gpucb':
        n_workers = int(10)
    else:
        n_workers = int(30)

    print configs
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
