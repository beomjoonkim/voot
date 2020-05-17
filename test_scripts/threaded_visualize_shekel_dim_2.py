import os
import sys
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
import argparse


def worker_p(config):
    command = 'python ./test_scripts/visualize_shekel_dim_2.py -seed ' \
              + str(config) \

    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    # python test_scripts/threaded_test_optimization_algorithms.py stovoo 10 1000 0,20 griewank 1 100,200,300,400,500,1000,5000 2,3,4,10,20,30,100 10,20,30,100
    # python test_scripts/threaded_test_optimization_algorithms.py voo 10 500 0,10 griewank 0 0 

    configs = range(100)
    n_workers = int(10)

    print configs
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
