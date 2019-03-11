import pickle
import argparse
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def get_result_dir(algo_name, dimension, obj_fcn, function_noise, algo_parameters):
    result_dir = './test_results/stochastic_function_optimization/' + str(obj_fcn) + '/dim_' + \
                 str(dimension) + '/noise_' + str(function_noise) + '/' + algo_name + '/ucb_' + str(
        algo_parameters['ucb']) + \
                 '/widening_' + str(algo_parameters['widening']) + '/'
    return result_dir


def plot_across_algorithms():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('-obj_fcn', type=str, default='ackley')
    parser.add_argument('-n_dim', type=int, default=10)
    parser.add_argument('-function_noise', type=float, default=200.0)
    args = parser.parse_args()

    algo_name = 'stovoo'

    widening_values = [2, 3, 4, 10, 20, 30, 100]
    ucb_values = [100.0, 200.0, 300.0, 400.0, 500.0, 1000.0, 5000.0]

    best_mean_value = -np.inf
    for widening_value in widening_values:
        for ucb in ucb_values:
            algo_parameters = {'ucb': ucb, 'widening': widening_value}
            fdir = get_result_dir(algo_name, dimension=args.n_dim, obj_fcn=args.obj_fcn,
                                  function_noise=args.function_noise, algo_parameters=algo_parameters)

            noise_level_max_values = []
            for fin in os.listdir(fdir):
                result = pickle.load(open(fdir + fin, 'r'))
                max_value = result['max_ys'][0][-1]
                noise_level_max_values.append(max_value)

            print "UCB %d, Widening value %.2f, performance (mean,var): %.2f %.2f" % \
                  (ucb, widening_value, np.mean(noise_level_max_values), np.std(noise_level_max_values))

            if np.mean(noise_level_max_values) >  best_mean_value:
                best_mean_value = np.mean(noise_level_max_values)
                best_configuration = (widening_value, ucb)
    print best_mean_value, best_configuration

if __name__ == '__main__':
    plot_across_algorithms()
