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
  plt.xlabel(xlabel,fontsize=14,fontweight='bold')
  plt.ylabel(ylabel,fontsize=14,fontweight='bold')
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  print 'Saving figure ',fname+'.png'
  plt.savefig(fname+'.png',dpi=100,format='png')

def get_stripstream_results(domain_name):
    if domain_name == 'convbelt':
        result_dir = './test_results/convbelt_results/stripstream/'
    else:
        result_dir = './test_results/convbelt_results/stripstream/'
    search_times = []
    success = []
    for fin in os.listdir(result_dir):
        if fin.find('.pkl') == -1: 
            continue
        try:
          result = pickle.load(open(result_dir+fin,'r'))
        except:
          print fin
        is_success = result['plan'] is not None
        success.append(is_success)
        if is_success:
          search_times.append(result['search_time'])
    print "stripstream time and success rate:"
    print np.array(search_times).mean()
    print np.array(success).mean()
    print len(search_times)


def get_result_dir(domain_name, algo_name):
    if algo_name.find('voo') != -1:
        epsilon = algo_name.split('_')[1]
        algo_name = algo_name.split('_')[0]
        rootdir = '/home/beomjoon/Dropbox (MIT)/braincloud/gtamp_results/test_results/'
    else:
        rootdir = '/home/beomjoon/Dropbox (MIT)/braincloud/gtamp_results/test_results/'

    if domain_name == 'convbelt':
        result_dir = rootdir+'/convbelt_results/uct_0.0_widening_0.5_'
    elif domain_name == 'namo':
        result_dir = rootdir+'/namo_results/uct_0.0_widening_0.5_'
    else:
        return -1

    result_dir += algo_name +'/'
    if algo_name.find('voo')!=-1:
        result_dir += 'eps_'+ str(epsilon)+'/'
    return result_dir


def get_mcts_results(domain_name, algo_name):
    result_dir = get_result_dir(domain_name, algo_name)
    search_times = []
    success = []
    search_rwd_times = []
    for fin in os.listdir(result_dir):
        if fin.find('.pkl') == -1: 
            continue
        if algo_name == 'voo':
            result = pickle.load(open(result_dir+fin,'r'))
        else:
            result = pickle.load(open(result_dir+fin,'r'))
        search_rwd_times.append(result['search_time'])

        if domain_name=='convbelt':
            is_success = result['plan'] is not None
            success.append(is_success)
            if is_success:
                search_times.append(result['search_time'][-1][0])
        else:
            is_success = result['search_time']['namo'][-1][-1]
            success.append(is_success)
            if is_success:
              search_times.append(result['search_time']['namo'][-1][0])

    print "mcts time and success rate:"
    print np.array(search_times).mean()
    print np.array(success).mean()
    print len(success)
    return search_rwd_times


def get_max_rwds_wrt_time(search_rwd_times):
    max_time = 310
    organized_times = range(10, max_time, 10)

    all_episode_data = []
    for rwd_time in search_rwd_times:
        episode_max_rwds_wrt_organized_times = []
        for organized_time in organized_times:
            episode_times = np.array(rwd_time)[:, 0]
            episode_rwds = np.array(rwd_time)[:, 1]
            idxs = episode_times < organized_time
            if np.any(idxs):
                max_rwd = np.max(episode_rwds[idxs])
            else:
                max_rwd = 0
            episode_max_rwds_wrt_organized_times.append(max_rwd)
        all_episode_data.append(episode_max_rwds_wrt_organized_times)

    return np.array(all_episode_data),organized_times


def get_max_rwds_wrt_time_namo(search_rwd_times):
    max_time = 510
    organized_times = range(10, max_time, 10)

    all_episode_data = []
    for rwd_time in search_rwd_times:
        episode_max_rwds_wrt_organized_times = []
        for organized_time in organized_times:
            episode_times = np.array(rwd_time['namo'])[:, 0]
            episode_rwds = np.array(rwd_time['namo'])[:, 1]
            idxs = episode_times < organized_time
            if np.any(idxs):
                max_rwd = np.max(episode_rwds[idxs])
            else:
                max_rwd = 0
            episode_max_rwds_wrt_organized_times.append(max_rwd)
        all_episode_data.append(episode_max_rwds_wrt_organized_times)

    return np.array(all_episode_data), organized_times


def main():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-domain', type=str, default='convbelt')
    parser.add_argument('-planner', type=str, default='stripstream')
    args = parser.parse_args()
    algo_names = ['voo_0.001', 'voo_0.01', 'voo_0.05', 'voo_0.08', 'voo_0.1', 'voo_0.3', 'voo_0.5', 'voo_0.7' ]
    #algo_names = ['voo_0.01', 'voo_0.3', 'voo_0.9','voo_0.001']
    #algo_names = ['voo_0.3', 'unif']

    if args.domain == 'namo':
        algo_names = ['unif', 'voo_0.001', 'voo_0.05', 'voo_0.7', 'voo_0.3']
    else:
        algo_names = ['unif', 'voo_0.001', 'voo_0.05', 'voo_0.7', 'voo_0.3']
        #algo_names = ['unif', 'voo_0.3']



    color_dict = pickle.load(open('./plotters/color_dict.p', 'r'))
    color_names = color_dict.keys()[1:]

    for algo_idx, algo in enumerate(algo_names):
        search_rwd_times = get_mcts_results(args.domain, algo)
        if args.domain == 'namo':
            search_rwd_times, organized_times = get_max_rwds_wrt_time_namo(search_rwd_times)
        else:
            search_rwd_times, organized_times = get_max_rwds_wrt_time(search_rwd_times)
        plot = sns.tsplot(search_rwd_times, organized_times, ci=95, condition=algo, color=color_dict[color_names[algo_idx]])
    savefig('Time (s)', 'Average rewards', fname='./plotters/'+args.domain)



if __name__ == '__main__':
    main()
