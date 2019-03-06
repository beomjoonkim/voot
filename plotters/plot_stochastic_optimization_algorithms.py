import pickle
import argparse
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def savefig(xlabel, ylabel, fname=''):
    plt.legend(loc='best', prop={'size': 13})
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    print 'Saving figure ', fname + '.png'
    plt.savefig(fname + '.png', dpi=100, format='png')


def get_result_dir(algo_name, dimension, obj_fcn):
    if obj_fcn != 'shekel':
        result_dir = './test_results/function_optimization/' + str(obj_fcn) + '/dim_' + str(
            dimension) + '/' + algo_name + '/'
    else:
        result_dir = './test_results/function_optimization/' + 'dim_' + str(dimension) + '/' + algo_name + '/'
        if algo_name == 'gpucb' and dimension == 10:
            result_dir = './test_results/function_optimization/' + 'dim_' + str(
                dimension) + '/' + algo_name + '/' + 'n_eval_200/'
    result_dir = './test_results/function_optimization/' + str(obj_fcn) + '/dim_' + str(
        dimension) + '/' + algo_name + '/'
    return result_dir


def get_results(algo_name, dimension, obj_fcn):
    result_dir = get_result_dir(algo_name, dimension, obj_fcn)
    search_times = []
    max_y_values = []
    time_takens = []
    for fin in os.listdir(result_dir):
        #for fin in os.listdir('./test_results//function_optimization/shekel/'+'dim_'+str(dimension)+'/gpucb/'):
        if fin.find('.pkl') == -1:
            continue
        result = pickle.load(open(result_dir + fin, 'r'))
        max_ys = np.array(result['max_ys'])
        if algo_name == 'doo':
            if obj_fcn != 'griewank':
                idxs = [0, 4, 10, 11, 12]
                optimal_epsilon_idx = np.argmax(max_ys[idxs, -1])
            else:
                optimal_epsilon_idx = np.argmax(max_ys[:, -1])
        else:
            optimal_epsilon_idx = np.argmax(max_ys[:, -1])
        max_y = max_ys[optimal_epsilon_idx, :]
        if len(max_y) < 500:
            continue
        if dimension == 2 and obj_fcn == 'shekel':
            max_y_values.append(max_y[:100])
            time_takens.append(result['time_takens'][optimal_epsilon_idx][:100])
        else:
            max_y_values.append(max_y)

            #time_takens.append(result['time_takens'][optimal_epsilon_idx])
    print 'number of functions tested ', len(max_y_values)
    return np.array(max_y_values)#, np.array(time_takens)


def get_max_rwds_wrt_time(search_rwd_times):
    max_time = 10000
    organized_times = range(100, max_time, 100)

    all_episode_data = []
    for rwd_time in search_rwd_times:
        episode_max_rwds_wrt_organized_times = []
        for organized_time in organized_times:
            if isinstance(rwd_time, dict):
                rwd_time_temp = rwd_time['namo']
                episode_times = np.array(rwd_time_temp)[:, 0]
                episode_rwds = np.array(rwd_time_temp)[:, 2]
            else:
                episode_times = np.array(rwd_time)[:, 0]
                episode_rwds = np.array(rwd_time)[:, 2]
            idxs = episode_times < organized_time
            if np.any(idxs):
                max_rwd = np.max(episode_rwds[idxs])
            else:
                max_rwd = 0
            episode_max_rwds_wrt_organized_times.append(max_rwd)
        all_episode_data.append(episode_max_rwds_wrt_organized_times)

    return np.array(all_episode_data), organized_times


def get_max_rwds_wrt_samples(search_rwd_times):
    organized_times = range(10, 1000, 10)

    all_episode_data = []
    for rwd_time in search_rwd_times:
        episode_max_rwds_wrt_organized_times = []
        for organized_time in organized_times:
            if isinstance(rwd_time, dict):
                rwd_time_temp = rwd_time['namo']
                episode_times = np.array(rwd_time_temp)[:, 1]
                # episode_rwds = np.array(rwd_time_temp)[:, -1]
                episode_rwds = np.array(rwd_time_temp)[:, 2]
            else:
                episode_times = np.array(rwd_time)[:, 1]
                episode_rwds = np.array(rwd_time)[:, 2]
            idxs = episode_times <= organized_time
            if np.any(idxs):
                max_rwd = np.max(episode_rwds[idxs])
            else:
                max_rwd = 0
            episode_max_rwds_wrt_organized_times.append(max_rwd)
        all_episode_data.append(episode_max_rwds_wrt_organized_times)
    return np.array(all_episode_data), organized_times


def plot_across_algorithms():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('-ucb', type=float, default=1.0)
    parser.add_argument('-widening_parameter', type=float, default=0.8)
    parser.add_argument('-problem_idx', type=int, default=0)
    parser.add_argument('-algo_name', type=str, default='voo')
    parser.add_argument('-obj_fcn', type=str, default='ackley')
    parser.add_argument('-dim_x', type=int, default=20)
    parser.add_argument('-n_fcn_evals', type=int, default=500)
    parser.add_argument('-stochastic_objective', action='store_true', default=False)
    parser.add_argument('-function_noise', type=float, default=10)
    args = parser.parse_args()

    n_dim = args.dim_x
    algo_names = ['stovoo']

    # x-axis: different noise level
    # y-axis: mean values of different widening values

    widening_values = [1.0, 10.0, 100.0, 200.0, 500.0]
    noise_levels = [100.0, 200.0, 300.0, 400.0,500.0,600.0,700.0,800.0,900.0,1000.0]

    for noise_level in noise_levels:
        for widening_value in widening_values:
            fdir = './test_results/stochastic_function_optimization/ackley/noise_'+str(noise_level) \
                   +'/ucb_5000.0/widening_'+str(widening_value) + '/dim_10/stovoo/'

            noise_level_max_values = []
            for fin in os.listdir(fdir):
                result = pickle.load(open(fdir+fin,'r'))
                #max_value = result['max_ys'][0][-1]
                max_value = result['best_arm_value']
                noise_level_max_values.append(max_value)

            print "Noise level %d, Widening value %d, performance (mean,var): %.2f %.2f" % (noise_level,widening_value,
                                                                                            np.mean(noise_level_max_values),
                                                                                            np.std(noise_level_max_values))




if __name__ == '__main__':
    plot_across_algorithms()
