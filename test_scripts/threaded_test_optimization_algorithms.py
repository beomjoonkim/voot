import os
import sys
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
import argparse


def worker_p(config):
    n_iter = config[0]
    problem_idx = config[1]
    algo_name = config[2]
    dim = config[3]
    obj_fcn = config[4]
    stochastic_objective = config[5]
    ucb = config[6]
    widening_parameter = config[7]
    noise = config[8]

    command = 'python ./test_scripts/test_optimization_algorithms.py -problem_idx ' \
              + str(problem_idx) \
              + ' -algo_name ' + algo_name \
              + ' -dim_x ' + str(dim) \
              + ' -n_fcn_evals ' + str(n_iter) \
              + ' -obj_fcn ' + str(obj_fcn) \
              + ' -widening_parameter ' + str(widening_parameter) \
              + ' -function_noise ' + str(noise)

    if stochastic_objective:
        command += ' -stochastic_objective'
        command += ' -ucb ' + str(ucb)

    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    # python test_scripts/threaded_test_optimization_algorithms.py stovoo 10 1000 0,20 griewank 1 100,200,300,400,500,1000,5000 2,3,4,10,20,30,100 10,20,30,100
    algo_name = sys.argv[1]
    dim = int(sys.argv[2])
    n_iter = sys.argv[3]
    pidxs = sys.argv[4].split(',')
    pidxs = range(int(pidxs[0]), int(pidxs[1]))
    obj_fcn = sys.argv[5]
    stochastic_objective = int(sys.argv[6])
    ucbs = sys.argv[7].split(',')
    ucbs = [float(ucb) for ucb in ucbs] 

    configs= []
    if stochastic_objective:
        widening_parameter = sys.argv[8].split(',')
        widening_parameters = [int(w) for w in widening_parameter]

        function_noise = sys.argv[9].split(',')
        function_noises = [int(f) for f in function_noise]
        for function_noise in function_noises:
          for ucb in ucbs:
            for widening_parameter in widening_parameters:
                for t in pidxs:
                    configs.append([n_iter, t, algo_name, dim, obj_fcn, stochastic_objective, ucb, widening_parameter,
                                    function_noise])
    else:
        for t in pidxs:
            configs.append([n_iter, t, algo_name, dim, obj_fcn, stochastic_objective, 0, 0, 0])

    if algo_name == 'gpucb':
        n_workers = int(3)
    else:
        n_workers = int(30)

    print configs
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
