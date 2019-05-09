import pickle
import argparse
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def savefig(xlabel, ylabel, fname=''):
    plt.legend(loc='best', prop={'size': 13})
    #plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    #plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    print 'Saving figure ', fname + '.png'
    plt.savefig(fname + '.png', dpi=100, format='png')


def get_result_dir(algo_name, dimension, obj_fcn):
    ROOTDIR = '/home/beomjoon/Dropbox (MIT)/braincloud/gtamp_results/'
    result_dir = ROOTDIR+'/function_optimization/' + str(obj_fcn) + '/dim_' + str(
        dimension) + '/' + algo_name + '/'
    return result_dir


def get_results(algo_name, dimension, obj_fcn):
    result_dir = get_result_dir(algo_name, dimension, obj_fcn)
    max_y_values = []

    try:
        result_files = os.listdir(result_dir)
    except OSError:
        return None

    for fin in result_files:
        if fin.find('.pkl') == -1:
            continue
        result = pickle.load(open(result_dir + fin, 'r'))
        max_ys = np.array(result['max_ys'])
        if algo_name == 'doo':
            optimal_epsilon_idx = np.argmax(max_ys[:, -1])
        else:
            optimal_epsilon_idx = np.argmax(max_ys[:, -1])
        max_y = max_ys[optimal_epsilon_idx, :]
        if len(max_y) < 500:
            continue
        else:
            max_y_values.append(max_y)
        print fin, len(max_y_values[-1]), max_y[-1], optimal_epsilon_idx

    print 'number of functions tested ', len(max_y_values)
    return np.array(max_y_values)


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
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-dim', type=int, default=3)
    parser.add_argument('-obj_fcn', type=str, default='shekel')
    args = parser.parse_args()
    n_dim = args.dim

    algo_names = ['gpucb', 'soo',  'voo', 'doo', 'uniform']
    color_dict = pickle.load(open('./plotters/color_dict.p', 'r'))
    color_names = color_dict.keys()
    color_dict[color_names[0]] = [0., 0.5570478679, 0.]
    color_dict[color_names[1]] = [0, 0, 0]
    color_dict[color_names[2]] = [1, 0, 0]
    color_dict[color_names[3]] = [0, 0, 1]
    color_dict[color_names[4]] = [0.8901960784313725, 0.6745098039215687, 0]

    ga_color = [0.2, 0.9, 0.1]
    ga_color = 'magenta'
    if args.obj_fcn != 'shekel':
        sns.tsplot([0]*2000, range(2000), ci=95, condition='Optimum', color='magenta')
    else:
        if n_dim == 3:
            plt.plot(range(2000), [4.89829] * 2000, linestyle='--', color=ga_color, label='GA_85k_evals')
        elif n_dim == 10:
            plt.plot(range(5000), [8.96] * 5000, linestyle='--', color=ga_color, label='GA_473k_evals')
        elif n_dim == 20:
            plt.plot(range(5000), [5.32] * 5000, linestyle='--', color=ga_color, label='GA_750k_evals')

    """
    if args.obj_fcn == 'rastrigin':
        if n_dim == 10:
            plt.plot(range(5000), [-21] * 5000, linestyle='--', color=ga_color, label='GA_40k_evals')
        elif n_dim == 20:
            plt.plot(range(5000), [-47] * 5000, linestyle='--', color=ga_color, label='GA_100k_evals')
    """

    if args.dim == 3 or args.obj_fcn == 'griewank':
        n_samples = 500
    elif args.obj_fcn == 'rosenbrock':
        n_samples = 5000
    elif args.obj_fcn == 'shekel' and args.dim == 20:
        n_samples = 5000
    else:
        n_samples = 1000

    for algo_idx, algo in enumerate(algo_names):
        print algo
        search_rwd_times = get_results(algo, n_dim, args.obj_fcn)
        if search_rwd_times is None:
            continue

        search_rwd_times = search_rwd_times[:, 0:n_samples]
        n_samples_tested = search_rwd_times.shape[-1]
        if n_samples_tested < n_samples:
            sns.tsplot(search_rwd_times, range(n_samples_tested), ci=95, condition=algo.upper(), color=color_dict[color_names[algo_idx]])
        else:
            sns.tsplot(search_rwd_times, range(n_samples), ci=95, condition=algo.upper(), color=color_dict[color_names[algo_idx]])
        print algo, n_samples, np.mean(search_rwd_times[:, -1])
        print "===================="

    plt.xlim(0, n_samples)
    savefig('Number of function evaluations', 'Best function values',
            fname='./plotters/' + args.obj_fcn + '_fcn_optimization_' + str(args.dim))


if __name__ == '__main__':
    plot_across_algorithms()
