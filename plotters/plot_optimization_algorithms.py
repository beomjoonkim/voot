import pickle
import argparse
import os
import numpy as np

import matplotlib

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import socket


def savefig(xlabel,ylabel,fname=''):
  plt.legend(loc='best',prop={'size': 13})
  plt.xlabel(xlabel,fontsize=14, fontweight='bold')
  plt.ylabel(ylabel,fontsize=14, fontweight='bold')
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  print 'Saving figure ', fname+'.png'
  plt.savefig(fname+'.png', dpi=100, format='png')


def get_result_dir(algo_name, dimension):
    result_dir = './test_results/function_optimization/'+'dim_'+str(dimension)+'/'+algo_name+'/'
    return result_dir


def get_results(algo_name, dimension):
    result_dir = get_result_dir(algo_name, dimension)
    search_times = []
    max_y_values = []
    for fin in os.listdir(result_dir):
        if fin.find('.pkl') == -1:
            continue
        result = pickle.load(open(result_dir+fin, 'r'))
        max_ys = np.array(result['max_ys'])
        optimal_epsilon_idx = np.argmax(max_ys[:, -1])
        max_y = max_ys[optimal_epsilon_idx, :]
        max_y_values.append(max_y)
    return np.array(max_y_values)


def get_max_rwds_wrt_time(search_rwd_times):
    max_time = 10000
    organized_times = range(100, max_time, 100)

    all_episode_data = []
    for rwd_time in search_rwd_times:
        episode_max_rwds_wrt_organized_times = []
        for organized_time in organized_times:
            if isinstance(rwd_time,dict):
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

    return np.array(all_episode_data),organized_times


def get_max_rwds_wrt_samples(search_rwd_times):
    organized_times = range(10, 1000, 10)

    all_episode_data = []
    for rwd_time in search_rwd_times:
        episode_max_rwds_wrt_organized_times = []
        for organized_time in organized_times:
            if isinstance(rwd_time,dict):
                rwd_time_temp = rwd_time['namo']
                episode_times = np.array(rwd_time_temp)[:, 1]
                #episode_rwds = np.array(rwd_time_temp)[:, -1]
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
    parser.add_argument('-dim', type=int, default=2)
    args = parser.parse_args()
    n_dim = args.dim

    algo_names = ['voo', 'uniform', 'doo']

    color_dict = pickle.load(open('./plotters/color_dict.p', 'r'))
    color_names = color_dict.keys()[1:]

    for algo_idx, algo in enumerate(algo_names):
        print algo
        search_rwd_times = get_results(algo, n_dim)
        mask = np.ones(len(search_rwd_times), dtype=bool)
        #too_large = np.where(search_rwd_times[:, -1] > np.mean(search_rwd_times[:, -1]) + np.std(search_rwd_times[:, -1]))[0]
        #print search_rwd_times
        if n_dim == 2:
            mask[[14,36]] = False
        elif n_dim==20:
            mask[125] = False
        search_rwd_times = search_rwd_times[mask]

        n_samples = search_rwd_times.shape[-1]
        sns.tsplot(search_rwd_times, range(1, n_samples+1), ci=95, condition=algo, color=color_dict[color_names[algo_idx]])
        print  "===================="
    plt.show()
    savefig('Number of simulations', 'Average rewards', fname='./plotters/'+args.domain+'_w_'+args.w)


if __name__ == '__main__':
    plot_across_algorithms()
