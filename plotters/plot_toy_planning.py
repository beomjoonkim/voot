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


def get_result_dir(algo_name, mcts_parameters):
    if algo_name.find('voo') != -1:
        if algo_name.find('standard_uniform') != -1:
            sampling_mode = 'standard_uniform'
        elif algo_name.find('gaussian') != -1:
            sampling_mode = 'gaussian'
        else:
            sampling_mode = 'centered_uniform'
        epsilon = algo_name.split('_')[-1]
        algo_name = algo_name.split('_')[0]
    elif algo_name.find('unif') != -1:
        algo_name = 'unif'
    elif algo_name.find('randomized_doo') !=-1:
        epsilon = algo_name.split('randomized_doo')[1][1:]
        algo_name = 'randomized_doo'

    uct = 0
    widening_parameter = mcts_parameters.w
    mcts_iter = mcts_parameters.mcts_iter
    n_feasibility_checks = mcts_parameters.n_feasibility_checks
    addendum = mcts_parameters.add
    n_switch = mcts_parameters.n_switch

    rootdir = './test_results/'
    result_dir = rootdir + '/' + mcts_parameters.domain + '/mcts_iter_'+str(mcts_iter)+ \
                 '/uct_0.0'+'_widening_' + str(widening_parameter) + '_' + algo_name
    result_dir += '_n_feasible_checks_' + str(n_feasibility_checks)
    if n_switch != -1:
        result_dir += '_n_switch_' + str(n_switch)

    if mcts_parameters.use_max_backup:
        result_dir += '_max_backup_True'

    if addendum != '':
        result_dir += '_' + addendum + '/'
    else:
        result_dir += '/'

    if algo_name.find('voo') != -1:
        result_dir += '/sampling_mode/' + sampling_mode + '/'
        result_dir += 'counter_ratio_' + str(mcts_parameters.counter_ratio) + '/'
        result_dir += 'eps_' + str(epsilon) + '/'
    if algo_name.find('doo') != -1 or algo_name.find('gpucb') != -1:
        result_dir += 'eps_' + str(epsilon) + '/'
    #    result_dir += os.listdir(result_dir)[0] + '/' #  + 'c1_' + str(c1) + '/'
    print result_dir
    return result_dir


def get_mcts_results(algo_name, mcts_parameters):
    result_dir = get_result_dir(algo_name, mcts_parameters)

    domain_name = mcts_parameters.domain
    pidx = mcts_parameters.pidx
    search_times = []
    progress = []
    search_rwd_times = []
    max_rwds = []
    success_idxs = []
    success_rewards = []
    for fin in os.listdir(result_dir):
        if fin.find('pidx') == -1:
            continue
        sd = int(fin.split('_')[2])
        file_pidx = int(fin.split('_')[-1].split('.')[0])

        if file_pidx != pidx:
            continue
        if fin.find('.pkl') == -1:
            continue
        result = pickle.load(open(result_dir + fin, 'r'))
        search_time = np.array(result['search_time'])
        progress.append(search_time[-1, -1] == 0)
        if search_time[-1,-1] ==0:
            success_idx = np.where( search_time[:,-1]==0)[0][0]
            success_rewards.append(search_time[success_idx,2])

        print len(search_time), fin
        search_rwd_times.append(search_time)
        max_rwds.append(np.max(search_time[:, 2]))

    print 'progress', np.array(progress).mean()
    print 'success reward', np.mean(success_rewards)
    print 'n_tested', len(progress)
    """
    print "mcts time and success rate:"
    print 'time', np.array(search_times).mean()
    print 'ff solution',np.array(success_idxs).mean()
    print 'max_rwd mean', np.mean(max_rwds)
    print 'ff min score', np.min(success_rewards)
    print 'ff mean score', np.mean(success_rewards)
    print 'n', len(search_rwd_times)
    """
    return search_rwd_times, np.mean(max_rwds)


def get_max_rwds_wrt_time(search_rwd_times):
    max_time = 7000
    organized_times = range(10, max_time, 1)

    all_episode_data = []
    all_episode_progress_data = []
    for rwd_time in search_rwd_times:
        episode_max_rwds_wrt_organized_times = []
        episode_max_progress_wrt_organized_times=[]
        for organized_time in organized_times:
            episode_times = np.array(rwd_time)[:, 0]
            episode_rwds = np.array(rwd_time)[:, 2]
            idxs = episode_times < organized_time
            if np.any(idxs):
                max_rwd = np.max(episode_rwds[idxs])
            else:
                max_rwd = 0
            episode_progress = -np.array(rwd_time)[:, 3]
            max_progress = np.max(episode_progress[idxs])
            episode_max_rwds_wrt_organized_times.append(max_rwd)
            episode_max_progress_wrt_organized_times.append(max_progress)
        all_episode_data.append(episode_max_rwds_wrt_organized_times)
        all_episode_progress_data.append(episode_max_progress_wrt_organized_times)

    return np.array(all_episode_data), np.array(all_episode_progress_data), organized_times


def get_max_rwds_wrt_samples(search_rwd_times, n_evals):
    organized_times = range(n_evals)

    all_episode_data = []
    all_episode_progress_data = []
    for rwd_time in search_rwd_times:
        episode_max_rwds_wrt_organized_times = []
        episode_max_progress_wrt_organized_times=[]
        for organized_time in organized_times:
            episode_times = np.array(rwd_time)[:, 1]
            episode_rwds = np.array(rwd_time)[:, 2]
            episode_progress = -np.array(rwd_time)[:, 3]
            idxs = episode_times <= organized_time
            max_rwd = np.max(episode_rwds[idxs])
            max_progress = np.max(episode_progress[idxs])
            episode_max_rwds_wrt_organized_times.append(max_rwd)
            episode_max_progress_wrt_organized_times.append(max_progress)
        all_episode_data.append(episode_max_rwds_wrt_organized_times)
        all_episode_progress_data.append(episode_max_progress_wrt_organized_times)
    return np.array(all_episode_data), np.array(all_episode_progress_data), organized_times


def get_algo_name(raw_name):
    if raw_name.find('randomized_doo') !=-1:
        return raw_name
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
    parser.add_argument('-domain', type=str, default='minimum_displacement_removal_results')
    parser.add_argument('-w', type=float, default=5.0)
    parser.add_argument('-c1', type=int, default=1)
    parser.add_argument('-uct', type=float, default=0.0)
    parser.add_argument('-mcts_iter', type=int, default=1500)
    parser.add_argument('-n_feasibility_checks', type=int, default=50)
    parser.add_argument('-pidx', type=int, default=0)
    parser.add_argument('--p', action='store_true')
    parser.add_argument('-add', type=str, default='fullplanning')
    parser.add_argument('-n_switch', type=int, default=10)
    parser.add_argument('-use_max_backup', action='store_true', default=False)
    parser.add_argument('-counter_ratio', type=int, default=10)

    args = parser.parse_args()

    algo_names = ['randomized_doo_1.0', 'voo_0.3', 'unif']
    algo_names = [ 'voo_uniform_0.3', 'unif']
    algo_names = [ 'voo_uniform_0.3', 'unif']
    algo_names = [ 'voo_uniform_0.3', 'unif']

    if args.domain == 'minimum_displacement_removal_results':
        if args.n_feasibility_checks == 100:
            algo_names = ['randomized_doo_0.001', 'randomized_doo_0.01', 'randomized_doo_0.1', 'voo_gaussian_0.3', 'unif']
        elif args.n_feasibility_checks == 50:
            algo_names = ['randomized_doo_1.0', 'voo_uniform_0.5', 'unif']
    else:
        algo_names = ['randomized_doo_1.0', 'randomized_doo_0.1', 'voo_gaussian_0.3', 'voo_gaussian_0.5',
                      'voo_uniform_0.3', 'voo_uniform_0.5', 'unif']
    #algo_names = ['randomized_doo_1.0', 'randomized_doo_0.1', 'voo_gaussian_0.3', 'voo_gaussian_0.5',
    #              'voo_uniform_0.3', 'voo_uniform_0.5', 'unif']

    #algo_names = ['randomized_doo_1.0', 'randomized_doo_0.1', 'voo_gaussian_0.3', 'voo_gaussian_0.5',
    #              'voo_uniform_0.3', 'voo_uniform_0.5', 'unif']

    #algo_names = ['randomized_doo_0.001', 'randomized_doo_0.01','randomized_doo_0.1','randomized_doo_1.0']
    algo_names = ['randomized_doo_1.0', 'randomized_doo_0.1', 'voo_standard_uniform_0.3', 'voo_standard_uniform_0.5', 'unif']
    #algo_names = ['voo_standard_uniform_0.3', 'voo_standard_uniform_0.5', 'unif']
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
        try:
            search_rwd_times, max_rwd = get_mcts_results(algo, args)
        except OSError:
            print algo, "not found"
            continue
        search_rwd, search_progress, organized_times = get_max_rwds_wrt_samples(search_rwd_times, args.mcts_iter)
        #search_rwd, search_progress, organized_times = get_max_rwds_wrt_time(search_rwd_times)

        max_rwds.append(max_rwd)
        algo_name = get_algo_name(algo)

        if algo_name in color_dict.keys():
            color = color_dict[algo_name]
        else:
            color = np.random.random((1, 3))

        if args.p:
            sns.tsplot(search_progress[:, :args.mcts_iter], organized_times[:args.mcts_iter], ci=95, condition=algo_name,
                       color=color)
        else:
            sns.tsplot(search_rwd[:, :args.mcts_iter], organized_times[:args.mcts_iter], ci=95, condition=algo_name,
                       color=color)

    if args.domain == 'minimum_displacement_removal_results':
        domain_name = 'mdr'
    else:
        domain_name = 'cbelt'

    if args.p:
        plot_name = 'progress_toy_'+domain_name+ '_pidx_' + str(args.pidx) + '_w_' + str(args.w) + '_mcts_iter_' + str(args.mcts_iter) \
                    + "_uct_" + str(args.uct) + "_n_feasibility_checks_" + str(args.n_feasibility_checks)
    else:
        if args.pidx == 0:
            sns.tsplot([4.1]*len(organized_times[:args.mcts_iter]), organized_times[:args.mcts_iter],
                       ci=95, condition='Avg feasible reward', color='magenta')
        else:
            sns.tsplot([2.97]*len(organized_times[:args.mcts_iter]), organized_times[:args.mcts_iter],
                       ci=95, condition='Avg feasible reward', color='magenta')

        plot_name = 'reward_toy_'+domain_name + '_pidx_' + str(args.pidx) + '_w_' + str(args.w) + '_mcts_iter_' + str(args.mcts_iter) \
                        + "_uct_" + str(args.uct) + "_n_feasibility_checks_" + str(args.n_feasibility_checks) + '_use_max_backup_' + str(args.use_max_backup)
        if args.n_switch != -1:
            plot_name += "_n_switch_" + str(args.n_switch)

    if args.domain.find('minimum') != -1:
        if args.pidx == 0:
            plt.ylim(-2, 4.6)

    savefig('Number of simulations', 'Average rewards', fname='./plotters/' + args.add + '_toy_'+plot_name)


if __name__ == '__main__':
    plot_across_algorithms()
