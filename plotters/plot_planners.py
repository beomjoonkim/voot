import cPickle as pickle
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


def get_result_dir(domain_name, algo_name, widening_parameter, c1, n_feasibility_checks, mcts_iter):
    if algo_name.find('voo') != -1:
        epsilon = algo_name.split('_')[1]
        algo_name = algo_name.split('_')[0]
        rootdir = '/home/beomjoon/Dropbox (MIT)/braincloud/gtamp_results/test_results/'
        # rootdir = './test_results/'
    elif algo_name.find('unif') != -1:
        rootdir = '/home/beomjoon/Dropbox (MIT)/braincloud/gtamp_results/test_results/'
    elif algo_name.find('randomized_doo') !=-1:
        epsilon = algo_name.split('randomized_doo')[1][1:]
        algo_name = 'randomized_doo'
        #rootdir = '/home/beomjoon/Dropbox (MIT)/braincloud/gtamp_results/test_results/'

    if domain_name == 'convbelt':
        rootdir = '/home/beomjoon/Dropbox (MIT)/braincloud/gtamp_results/test_results//'
        result_dir = rootdir + '/convbelt_results/mcts_iter_' +str(mcts_iter)+'/uct_0.0_widening_' + str(widening_parameter) + '_'
    elif domain_name == 'namo':
        rootdir = '/home/beomjoon/Dropbox (MIT)/braincloud/gtamp_results/test_results//root_switching/no_infeasible_place/'
        result_dir = rootdir + '/namo_results/mcts_iter_500/uct_0.0_widening_' + str(widening_parameter) + '_'
        result_dir = rootdir + '/namo_results/mcts_iter_'+str(mcts_iter)+'/uct_0.0_widening_' + str(widening_parameter) + '_'
    else:
        return -1
    if algo_name.find('plaindoo') == -1:
        result_dir += algo_name
    result_dir += '_n_feasible_checks_' + str(n_feasibility_checks) + '/'
    if algo_name.find('voo') != -1 or algo_name.find('doo') != -1 or algo_name.find('gpucb') != -1:
        result_dir += 'eps_' + str(epsilon) + '/' + 'c1_' + str(c1) + '/'
    print result_dir
    return result_dir


def get_mcts_results(domain_name, algo_name, widening_parameter, c1, n_feasibility_checks,mcts_iter, pidx):
    result_dir = get_result_dir(domain_name, algo_name, widening_parameter, c1, n_feasibility_checks,mcts_iter)
    search_times = []
    success = []
    search_rwd_times = []
    max_rwds = []
    success_idxs = []
    for fin in os.listdir(result_dir):
        if fin.find('pidx') == -1:
            continue
        sd = int(fin.split('_')[2])
        file_pidx = int(fin.split('_')[-1].split('.')[0])

        if file_pidx != pidx:
            continue
        if fin.find('.pkl') == -1:
            continue
        if algo_name == 'voo':
            result = pickle.load(open(result_dir + fin, 'r'))
        else:
            try:
                result = pickle.load(open(result_dir + fin, 'r'))
            except:
                continue
        if domain_name == 'namo':
            assert isinstance(result['search_time'], dict)
        print result_dir+fin
        if domain_name == 'convbelt':
            is_success = result['plan'] is not None
            max_rwds.append( np.max(np.array(result['search_time'])[:,2]))
            print np.max(np.array(result['search_time'])[:, 2]), np.max(result['reward_list'])

            search_rwd_times.append(result['search_time'])
            #is_success = np.any(np.array(result['search_time'])[:, 2] >= 4)
            #if is_success:
            #    search_times.append(np.where(np.array(result['search_time'])[:, 2] >= 4)[0][0])

            # search_times.append(np.array(result['search_time'])[:,0][-1])
            success.append(is_success)
        else:
            search_rwd_times.append(result['search_time'])
            is_success = np.any(np.array(result['search_time']['namo'])[:,-1])
            max_rwds.append( np.max(np.array(result['search_time']['namo'])[:,2]))
            success.append(is_success)
            if is_success:
                success_idxs.append( np.where(np.array(result['search_time']['namo'])[:,-1])[0][0]+1 )
                search_times.append(result['search_time']['namo'][-1][0])

    print "mcts time and success rate:"
    print 'time', np.array(search_times).mean()
    print 'success', np.array(success).mean()
    print 'ff solution',np.array(success_idxs).mean()
    print 'max_rwd mean', np.mean(max_rwds)
    print 'max_rwd std', np.std(max_rwds)
    print 'max_rwd max', np.max(max_rwds)
    print 'n', len(success)
    return search_rwd_times, np.mean(max_rwds)


def get_max_rwds_wrt_time(search_rwd_times):
    max_time = 700
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


def get_algo_name(raw_name):
    if raw_name.find('randomized_doo') !=-1:
        return "RandDOO"
    elif raw_name.find('voo') != -1:
        return 'VOO'
    elif raw_name.find('unif') != -1:
        return "Uniform"
    else:
        raise ValueError


def plot_across_algorithms():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-domain', type=str, default='convbelt')
    parser.add_argument('-w', type=float, default=0.8)
    parser.add_argument('-c1', type=float, default=1.0)
    parser.add_argument('-mcts_iter', type=int, default=500)
    parser.add_argument('-n_feasibility_checks', type=int, default=50)
    parser.add_argument('-pidx', type=int, default=0)
    parser.add_argument('--t', action='store_true')

    args = parser.parse_args()
    widening_parameter = args.w

    if args.domain == 'namo':
        algo_names = ['randomized_doo_1.0', 'voo_0.3', 'unif' ]
        #algo_names = ['randomized_doo_1.0', 'voo_0.3' ]
    else:
        algo_names = ['randomized_doo_1.0', 'voo_0.3','unif']
        #algo_names = ['randomizeddoo_1.0', 'voo_0.3' ]

    color_dict = pickle.load(open('./plotters/color_dict.p', 'r'))
    color_names = color_dict.keys()[1:]
    color_names = color_dict.keys()
    color_dict[color_names[0]] = [0., 0.5570478679, 0.]
    color_dict['RandDOO'] = [0, 0, 0]
    color_dict['VOO'] = [1, 0, 0]
    color_dict['Uniform'] = [0, 0, 1]


    # DOO - black
    # VOO - red
    # Uniform - blue

    averages = []
    max_rwds = []
    for algo_idx, algo in enumerate(algo_names):
        print algo
        search_rwd_times, max_rwd = get_mcts_results(args.domain, algo, widening_parameter, args.c1,
                                                        args.n_feasibility_checks, args.mcts_iter, args.pidx)
        max_rwds.append(max_rwd)
        if args.t:
            search_rwd_times, organized_times = get_max_rwds_wrt_time(search_rwd_times)
        else:
            search_rwd_times, organized_times = get_max_rwds_wrt_samples(search_rwd_times)

        algo_name = get_algo_name(algo)
        sns.tsplot(search_rwd_times, organized_times, ci=95, condition=algo_name, color=color_dict[algo_name])
        print "===================="

    sns.tsplot([0.95*np.max(max_rwds)]*len(organized_times), organized_times, ci=95, condition='95% optimal', color='magenta')
    #if args.domain=='namo':
    #    plt.ylim([2, 4.5])

    if args.t:
        savefig('Times (s)', 'Average rewards', fname='./plotters/t_' + args.domain + '_w_' + str(args.w))
    else:
        savefig('Number of evaluations', 'Average rewards', fname='./plotters/' + args.domain + '_pidx_' + str(args.pidx))


if __name__ == '__main__':
    plot_across_algorithms()
