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


def get_result_dir(algo_name, dimension, obj_fcn, function_noise, algo_parameters):
    result_dir = './test_results/stochastic_function_optimization/' + str(obj_fcn) + '/dim_' + \
                 str(dimension) + '/noise_'+str(function_noise) + '/' + algo_name + '/ucb_'+str(algo_parameters['ucb']) +\
                 '/widening_' + str(algo_parameters['widening']) + '/'
    return result_dir


def get_results(algo_name, args, algo_parameters):
    obj_fcn = args.obj_fcn
    dimension = args.n_dim
    result_dir = get_result_dir(algo_name, dimension, obj_fcn, args.function_noise, algo_parameters)

    max_y_values = []
    time_takens = []
    for fin in os.listdir(result_dir):
        if fin.find('.pkl') == -1:
            continue
        result = pickle.load(open(result_dir + fin, 'r'))
        max_ys = np.array(result['max_ys'])
        optimal_epsilon_idx = np.argmax(max_ys[:, -1])
        max_y = max_ys[optimal_epsilon_idx, :]
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
    parser.add_argument('-obj_fcn', type=str, default='griewank')
    parser.add_argument('-n_dim', type=int, default=10)
    parser.add_argument('-function_noise', type=float, default=200.0)
    args = parser.parse_args()

    algo_names = ['stovoo', 'stovoo_with_N_eta', 'stosoo']
    if args.obj_fcn == 'ackley':
        if args.function_noise == 30:
            algo_parameters = {'stovoo': {'ucb': 100.0, 'widening': 2},
                               'stosoo': {'ucb': 1.0, 'widening': 1},
                               'stounif': {'ucb': 1.0, 'widening': 1}}
        elif args.function_noise == 0:
            algo_parameters = {'stovoo': {'ucb': 1.0, 'widening': 1},
                               'stosoo': {'ucb': 1.0, 'widening': 1},
                               'stounif': {'ucb': 1.0, 'widening': 1}}
        elif args.function_noise == 10:
            algo_parameters = {'stovoo': {'ucb': 200.0, 'widening': 4},
                               'stosoo': {'ucb': 1.0, 'widening': 1},
                               'stounif': {'ucb': 1.0, 'widening': 1}}
        elif args.function_noise == 100:
            algo_parameters = {'stovoo': {'ucb': 100.0, 'widening': 3},
                               'stosoo': {'ucb': 1.0, 'widening': 1},
                               'stounif': {'ucb': 1.0, 'widening': 1}}
        elif args.function_noise == 500:
            algo_parameters = {'stovoo': {'ucb': 400.0, 'widening': 10},
                               'stosoo': {'ucb': 1.0, 'widening': 1},
                               'stounif': {'ucb': 1.0, 'widening': 1}}
        elif args.function_noise == 1000:
            algo_parameters = {'stovoo': {'ucb': 500.0, 'widening': 10},
                               'stosoo': {'ucb': 1.0, 'widening': 1},
                               'stounif': {'ucb': 1.0, 'widening': 1}}
    elif args.obj_fcn == 'griewank':
        algo_parameters = {'stovoo': {'ucb': 100.0, 'widening': 0.3},
                           'stovoo_with_N_eta': {'ucb': 100.0, 'widening': 10.0},
                           'stosoo': {'ucb': 1.0, 'widening': 1.0},
                           'stounif': {'ucb': 1.0, 'widening': 1},
                           }

    # todo
    #   find the fastest-growing parameters

    color_dict = pickle.load(open('./plotters/color_dict.p', 'r'))
    color_names = color_dict.keys()
    color_dict[color_names[0]] = [0., 0.5570478679, 0.]
    color_dict[color_names[1]] = [0, 0, 0]
    color_dict[color_names[2]] = [1, 0, 0]
    color_dict[color_names[3]] = [0, 0, 1]
    color_dict[color_names[4]] = [0.8901960784313725, 0.6745098039215687, 0]

    sns.tsplot([0] * 2000, range(2000), ci=95, condition='Optimum', color='magenta')
    for algo_idx, algo_name in enumerate(algo_names):
        search_rwd_times = get_results(algo_name, args, algo_parameters[algo_name])
        n_samples = search_rwd_times.shape[-1]

        sns.tsplot(search_rwd_times, range(n_samples), ci=95, condition=algo_name.upper(),
                   color=color_dict[color_names[algo_idx]])

        print algo_name, np.mean(search_rwd_times,axis=0)[-1]
    plt.show()
    import pdb;pdb.set_trace()





if __name__ == '__main__':
    plot_across_algorithms()
