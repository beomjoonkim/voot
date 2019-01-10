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


def get_result_dir(domain_name, algo_name, widening_parameter):
    if algo_name.find('voo') != -1:
        epsilon = algo_name.split('_')[1]
        algo_name = algo_name.split('_')[0]
        rootdir = './test_results/'
        rootdir = '/home/beomjoon/Dropbox (MIT)/braincloud/gtamp_results/test_results/'
        rootdir = './test_results/'
    else:
        rootdir = '/home/beomjoon/Dropbox (MIT)/braincloud/gtamp_results/test_results/'
        rootdir = './test_results/'
        #if domain_name == 'namo':
        #    rootdir = './test_results/'

    """
        epsilon = algo_name.split('_')[1]
        algo_name = algo_name.split('_')[0]
        rootdir = '/home/beomjoon/Dropbox (MIT)/braincloud/gtamp_results/test_results/'
        if domain_name == 'namo':
            rootdir = './test_results/'
    else:
        if domain_name == 'namo':
            rootdir = '/home/beomjoon/Dropbox (MIT)/braincloud/gtamp_results/test_results/'
        else:
            rootdir = './test_results/'
    """

    if domain_name == 'convbelt':
        result_dir = rootdir+'/convbelt_results/mcts_iter_50/uct_0.0_widening_'+ str(widening_parameter)+'_'
        print result_dir
    elif domain_name == 'namo':
        result_dir = rootdir+'/namo_results/mcts_iter_50/uct_0.0_widening_' + str(widening_parameter)+'_'
    else:
        return -1

    result_dir += algo_name +'/'
    if algo_name.find('voo')!=-1:
        result_dir += 'eps_'+ str(epsilon)+'/'
    return result_dir


def get_mcts_results(domain_name, algo_name, widening_parameter):
    result_dir = get_result_dir(domain_name, algo_name, widening_parameter)
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
        if domain_name == 'namo':
            assert isinstance(result['search_time'], dict)

        if domain_name=='convbelt':
            is_success = result['plan'] is not None
            is_success = np.any(np.array(result['search_time'])[:,2] >= 4)
            #search_times.append( np.where(np.array(result['search_time'])[:,2]>=4)[0][0])
            search_times.append(np.array(result['search_time'])[:,0][-1])
            success.append(is_success)
        else:
            is_success = result['search_time']['namo'][-1][-1]
            success.append(is_success)
            if is_success:
              search_times.append(result['search_time']['namo'][-1][0])

    print "mcts time and success rate:"
    print 'time', np.array(search_times).mean()
    print 'success', np.array(success).mean()
    print 'n', len(success)
    return search_rwd_times


def get_max_rwds_wrt_time(search_rwd_times):
    max_time = 310
    organized_times = range(10, max_time, 10)

    all_episode_data = []
    for rwd_time in search_rwd_times:
        episode_max_rwds_wrt_organized_times = []
        for organized_time in organized_times:
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
    organized_times = range(55)

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


def plot_across_algorithms():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-domain', type=str, default='convbelt')
    parser.add_argument('-w', type=str, default='none')

    args = parser.parse_args()
    widening_parameter = args.w

    if args.domain == 'namo':
        algo_names = ['unif', 'voo_0.1', 'voo_0.2', 'voo_0.3']
    else:
        algo_names = ['unif', 'voo_0.01', 'voo_0.1', 'voo_0.2', 'voo_0.3', 'voo_0.4', 'voo_0.5']

    color_dict = pickle.load(open('./plotters/color_dict.p', 'r'))
    color_names = color_dict.keys()[1:]

    for algo_idx, algo in enumerate(algo_names):
        print algo
        try:
            search_rwd_times = get_mcts_results(args.domain, algo, widening_parameter)
        except:
            continue
        if args.domain == 'namo':
            search_rwd_times, organized_times = get_max_rwds_wrt_samples(search_rwd_times)
        else:
            search_rwd_times, organized_times = get_max_rwds_wrt_samples(search_rwd_times)
        plot = sns.tsplot(search_rwd_times, organized_times, ci=95, condition=algo, color=color_dict[color_names[algo_idx]])
        print  "===================="
    #plt.show()
    savefig('Number of simulations', 'Average rewards', fname='./plotters/'+args.domain+'_w_'+args.w)

def plot_across_widening_parameters():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-domain', type=str, default='convbelt')
    parser.add_argument('-algo_name', type=str, default='unif')

    args = parser.parse_args()
    widening_parameters = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    widening_parameters = [0.6,0.7,0.2,0.8]
    algo = args.algo_name

    color_dict = pickle.load(open('./plotters/color_dict.p', 'r'))
    color_names = color_dict.keys()[2:]
    for widening_idx, widening_parameter in enumerate(widening_parameters):
        print algo + str(widening_parameter)
        try:
            search_rwd_times = get_mcts_results(args.domain, algo, widening_parameter)
        except:
            continue
        if args.domain == 'namo':
            search_rwd_times, organized_times = get_max_rwds_wrt_samples(search_rwd_times)
        else:
            search_rwd_times, organized_times = get_max_rwds_wrt_samples(search_rwd_times)
        plot = sns.tsplot(search_rwd_times, organized_times, ci=95, condition=algo+'_'+str(widening_parameter),
                          color=color_dict[color_names[widening_idx]])
        print  "===================="
    # plt.show()
    savefig('Number of simulations', 'Average rewards', fname='./plotters/' + args.domain + '_algo_' + args.algo_name)


if __name__ == '__main__':
    plot_across_algorithms()
    #plot_across_widening_parameters()
