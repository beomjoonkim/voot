import pickle
import argparse
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def get_result_dir(algo_name, dimension, obj_fcn, function_noise, algo_parameters):
    result_dir = './test_results/stochastic_function_optimization/' + str(obj_fcn) + '/dim_' + \
                 str(dimension) + '/noise_' + str(function_noise) + '/' + algo_name + '/ucb_' + \
                 str(algo_parameters['ucb']) + '/widening_' + str(algo_parameters['widening']) + '/'
    return result_dir


def plot_across_algorithms():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('-obj_fcn', type=str, default='griewank')
    parser.add_argument('-n_dim', type=int, default=10)
    parser.add_argument('-function_noise', type=float, default=200.0)
    args = parser.parse_args()

    algo_name = 'stovoo_with_N_eta'
    #algo_name = 'stovoo'

    widening_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    widening_values = [0.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0]
    ucb_values = [100.0]

    mean_values = []
    configurations = []
    best_mean_value = -np.inf
    for widening_value in widening_values:
        for ucb in ucb_values:
            algo_parameters = {'ucb': ucb, 'widening': widening_value}
            fdir = get_result_dir(algo_name, dimension=args.n_dim, obj_fcn=args.obj_fcn,
                                  function_noise=args.function_noise, algo_parameters=algo_parameters)

            noise_level_max_values = []
            try:
                file_list = os.listdir(fdir)
            except:
                continue
            for fin in file_list:
                result = pickle.load(open(fdir + fin, 'r'))
                max_ys = np.array(result['max_ys'])
                try:
                    assert len(max_ys[:, -1]) == 5, 'You should try at least five different epsilon values'
                except:
                    print widening_value, ucb
                    continue
                optimal_epsilon_idx = np.argmax(max_ys[:, -1])
                max_value = max_ys[optimal_epsilon_idx][-1]
                noise_level_max_values.append(max_value)

            print "UCB %d, Widening value %.2f, performance (mean,var): %.2f %.2f" % \
                  (ucb, widening_value, np.mean(noise_level_max_values), np.std(noise_level_max_values))

            if np.mean(noise_level_max_values) > best_mean_value:
                best_mean_value = np.mean(noise_level_max_values)
                best_configuration = (widening_value, ucb)

            mean_values.append(np.mean(noise_level_max_values))
            configurations.append((widening_value, ucb))

    print best_mean_value, best_configuration

    # todo: there is also the rate at which this was achieved
    sorted_idxs = np.argsort(mean_values)
    print np.array(mean_values)[sorted_idxs]
    print np.array(configurations)[sorted_idxs]


if __name__ == '__main__':
    plot_across_algorithms()
