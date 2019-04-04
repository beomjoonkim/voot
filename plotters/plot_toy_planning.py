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


#def get_result_dir(domain_name, algo_name, widening_parameter, c1, n_feasibility_checks, mcts_iter):
def get_result_dir(algo_name, mcts_parameters):
    if algo_name.find('voo') != -1:
        sampling_mode = algo_name.split('_')[1]
        epsilon = algo_name.split('_')[2]
        algo_name = algo_name.split('_')[0]
    elif algo_name.find('unif') != -1:
        algo_name = 'unif'
    elif algo_name.find('randomized_doo') !=-1:
        epsilon = algo_name.split('randomized_doo')[1][1:]
        algo_name = 'randomized_doo'

    uct = 0
    widening_parameter = mcts_parameters.w
    mcts_iter = mcts_parameters.mcts_iter
    rootdir = './test_results/'
    result_dir = rootdir + '/minimum_displacement_removal_results/mcts_iter_'+str(mcts_iter)+ \
                 '/uct_0.0'+'_widening_' + str(widening_parameter) + '_'

    n_feasibility_checks = mcts_parameters.n_feasibility_checks
    c1 = mcts_parameters.c1
    if algo_name.find('plaindoo') == -1:
        result_dir += algo_name
    result_dir += '_n_feasible_checks_' + str(n_feasibility_checks) + '/'

    if algo_name.find('voo') != -1:
        result_dir += '/sampling_mode/' + sampling_mode + '/'
        result_dir += 'eps_' + str(epsilon) + '/'
    if algo_name.find('doo') != -1 or algo_name.find('gpucb') != -1:
        result_dir += 'eps_' + str(epsilon) + '/'
        result_dir += os.listdir(result_dir)[0] + '/' #  + 'c1_' + str(c1) + '/'
    print result_dir
    return result_dir


def get_mcts_results(algo_name, mcts_parameters):
    result_dir = get_result_dir(algo_name, mcts_parameters)

    domain_name = mcts_parameters.domain
    pidx = mcts_parameters.pidx
    search_times = []
    success = []
    search_rwd_times = []
    max_rwds = []
    success_idxs = []
    success_rewards = []
    for fin in os.listdir(result_dir):
        if domain_name == 'namo':
            if fin.find('pidx') == -1:
                continue
            sd = int(fin.split('_')[2])
            file_pidx = int(fin.split('_')[-1].split('.')[0])

            if file_pidx != pidx:
                continue
            if fin.find('.pkl') == -1:
                continue
        result = pickle.load(open(result_dir + fin, 'r'))
        search_time = np.array(result['search_time'])[0:1000,:]

        print len(search_time), fin
        search_rwd_times.append(search_time[0:1000,:])
        max_rwds.append(np.max(search_time[:, 2]))

    """
    print "mcts time and success rate:"
    print 'time', np.array(search_times).mean()
    print 'success', np.array(success).mean()
    print 'ff solution',np.array(success_idxs).mean()
    print 'max_rwd mean', np.mean(max_rwds)
    print 'ff min score', np.min(success_rewards)
    print 'ff mean score', np.mean(success_rewards)
    print 'n', len(search_rwd_times)
    """
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
    organized_times = range(30)

    all_episode_data = []
    for rwd_time in search_rwd_times:
        episode_max_rwds_wrt_organized_times = []
        for organized_time in organized_times:
            episode_times = np.array(rwd_time)[:, 1]
            episode_rwds = -np.array(rwd_time)[:, 3]
            idxs = episode_times <= organized_time
            max_rwd = np.max(episode_rwds[idxs])
            episode_max_rwds_wrt_organized_times.append(max_rwd)
        all_episode_data.append(episode_max_rwds_wrt_organized_times)
    return np.array(all_episode_data), organized_times


def get_algo_name(raw_name):
    if raw_name.find('randomized_doo') !=-1:
        return "RandDOOT"
    elif raw_name.find('voo') != -1:
        return raw_name
        return 'VOOT'
    elif raw_name.find('unif') != -1:
        return "UniformT"
    else:
        raise ValueError


def plot_across_algorithms():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-domain', type=str, default='mdr')
    parser.add_argument('-w', type=float, default=1.0)
    parser.add_argument('-c1', type=int, default=1)
    parser.add_argument('-uct', type=float, default=0.0)
    parser.add_argument('-mcts_iter', type=int, default=30)
    parser.add_argument('-n_feasibility_checks', type=int, default=50)
    parser.add_argument('-pidx', type=int, default=0)
    parser.add_argument('--t', action='store_true')

    args = parser.parse_args()

    algo_names = ['randomized_doo_1.0', 'voo_0.3', 'unif']
    algo_names = ['voo_uniform_0.3','unif']
    #algo_names = ['voo_0.3', 'unif']

    color_dict = pickle.load(open('./plotters/color_dict.p', 'r'))
    color_names = color_dict.keys()
    color_dict[color_names[0]] = [0., 0.5570478679, 0.]
    color_dict['RandDOOT'] = [0, 0, 0]
    color_dict['VOOT'] = [1, 0, 0]
    color_dict['UniformT'] = [0, 0, 1]

    # DOO - black
    # VOO - red
    # Uniform - blue

    averages = []
    max_rwds = []
    for algo_idx, algo in enumerate(algo_names):
        print algo
        #pkl_fname = './plotters/planning_results/search_rwd_times_pkl_files/'+args.domain+'_pidx_'+str(args.pidx)+'_'+algo+'.pkl'
        #if os.path.isfile(pkl_fname):
        #    search_rwd_times, organized_times, max_rwd = pickle.load(open(pkl_fname,'r'))
        #else:
        search_rwd_times, max_rwd = get_mcts_results(algo, args)
        search_rwd_times, organized_times = get_max_rwds_wrt_samples(search_rwd_times)
        #pickle.dump((search_rwd_times, organized_times, max_rwd), open(pkl_fname, 'wb'))

        max_rwds.append(max_rwd)
        algo_name = get_algo_name(algo)

        if algo_name in color_dict.keys():
            color = color_dict[algo_name]
        else:
            color = np.random.random((1, 3))
        sns.tsplot(search_rwd_times[:, :args.mcts_iter], organized_times[:args.mcts_iter], ci=95, condition=algo_name,
                   color=color)
        print "===================="

        sns.tsplot([0.962]*len(organized_times[:args.mcts_iter]), organized_times[:args.mcts_iter],
                   ci=95, condition='Avg feasible reward', color='magenta')

    plot_name = args.domain + '_pidx_' + str(args.pidx) + '_w_' + str(args.w) + '_mcts_iter_' + str(args.mcts_iter) \
                    + "_uct_" + str(args.uct)
    savefig('Number of simulations', 'Average rewards', fname='./plotters/toy_'+plot_name)


if __name__ == '__main__':
    plot_across_algorithms()
