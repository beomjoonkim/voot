import cPickle as pickle
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import glob

import plot_rl

def savefig(xlabel, ylabel, fname=''):
    plt.legend(loc='best', prop={'size': 13})
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    print 'Saving figure ', fname + '.png'
    plt.tight_layout()
    plt.savefig(fname + '.png', dpi=100, format='png')


def get_best_hyperparameter_dir(result_dir, problem_idx):
    best_dir = None
    best_rwd_among_all_setups = -np.inf
    fdirs = glob.glob(result_dir)

    print 'n hyper-parameters tested', len(fdirs)
    for fidx, fdir in enumerate(fdirs):
        last_rwds = []
        listfiles = os.listdir(fdir)
        if len(listfiles) < 10:
            continue
        for fin in listfiles:
            if fin.find('pidx') == -1:
                continue

            sd = int(fin.split('_')[2])
            file_problem_idx = int(fin.split('_')[-1].split('.')[0])

            if file_problem_idx != problem_idx:
                print "file problem idx does not match the given problem idx"
                continue
            try:
                result = pickle.load(open(fdir + fin, 'r'))
            except EOFError:
                print "EOF error", fin
                continue
            search_time = np.array(result['search_time'])
            max_rwd = search_time[-1, -2]

            last_rwds.append(max_rwd)
        avg_last_rwds = np.mean(last_rwds)
        if avg_last_rwds > best_rwd_among_all_setups:
            best_dir = fdir
            best_rwd_among_all_setups = avg_last_rwds

    if best_dir is None:
        print 'Dir empty', result_dir
    print best_dir
    return best_dir, best_rwd_among_all_setups


def get_result_dir(algo_name, mcts_parameters):
    voo_sampling_mode = ''
    if algo_name.find('voo') != -1:
        voo_sampling_mode = algo_name[4:]
        algo_name = 'voo'
    elif algo_name.find('unif') != -1:
        algo_name = 'unif'
    elif algo_name.find('randomized_doo') != -1:
        algo_name = 'randomized_doo'
    elif algo_name.find('doo') != -1:
        algo_name = 'doo'

    widening_parameter = mcts_parameters.w
    mcts_iter = mcts_parameters.mcts_iter
    n_feasibility_checks = mcts_parameters.n_feasibility_checks
    addendum = mcts_parameters.add
    n_switch = mcts_parameters.n_switch

    if 'mdr' in mcts_parameters.domain:
        domain = 'minimum_displacement_removal'
    else:
        domain = mcts_parameters.domain

    rootdir = './test_results/'
    result_dir = rootdir + '/' + domain + '_results/mcts_iter_' + str(mcts_iter)
    if algo_name.find('pw') != -1:
        result_dir += '/pw_methods/uct_*_widening_*_unif'
    else:
        result_dir += '/uct_0.0' + '_widening_' + str(widening_parameter) + '_' + algo_name
    result_dir += '_n_feasible_checks_' + str(n_feasibility_checks)

    if n_switch != -1:
        result_dir += '_n_switch_' + str(n_switch)

    if mcts_parameters.use_max_backup:
        result_dir += '_max_backup_True'
    else:
        result_dir += '_max_backup_False'

    if mcts_parameters.pick_switch:
        result_dir += '_pick_switch_True'
    else:
        result_dir += '_pick_switch_False'
    result_dir += '_n_actions_per_node_' + str(mcts_parameters.n_actions_per_node)

    if mcts_parameters.domain.find('synthetic') != -1:
        if mcts_parameters.domain.find('rastrigin') != -1:
            if mcts_parameters.problem_idx == 1 or algo_name.find('pw') != -1:
                result_dir += '_value_threshold_-50.0'  # + str(mcts_parameters.value_threshold)
        elif mcts_parameters.domain.find('shekel') != -1 and algo_name.find('pw') != -1:
            result_dir += '_value_threshold_' + str(mcts_parameters.value_threshold)

    if addendum != '':
        if algo_name != 'pw':
            result_dir += '_' + addendum + '/'
        else:
            result_dir += '_pw_reevaluates_infeasible/'
    else:
        result_dir += '/'

    problem_idx = mcts_parameters.problem_idx

    if algo_name == 'voo':
        result_dir += '/sampling_mode/' + voo_sampling_mode + '/counter_ratio_1/eps_*/'
    elif algo_name.find('doo') != -1:
        result_dir += '/eps_*/'
    result_dir, best_rwd = get_best_hyperparameter_dir(result_dir, problem_idx)

    print 'Best rwd', best_rwd
    return result_dir


def get_mcts_results(algo_name, mcts_parameters):
    result_dir = get_result_dir(algo_name, mcts_parameters)
    problem_idx = mcts_parameters.problem_idx
    progress = []
    search_rwd_times = []
    max_rwds = []
    success_rewards = []

    try:
        list_dir = os.listdir(result_dir)
    except TypeError:
        print result_dir
        sys.exit()

    for fin in list_dir:
        if fin.find('pidx') == -1:
            # print "Continuing"
            continue
        sd = int(fin.split('_')[2])
        file_problem_idx = int(fin.split('_')[-1].split('.')[0])

        if file_problem_idx != problem_idx:
            # print "Continuing"
            continue
        try:
            result = pickle.load(open(result_dir + fin, 'r'))
        except EOFError:
            print "EOF error", fin
            continue
        search_time = np.array(result['search_time'])

        if mcts_parameters.domain.find('convbelt') != -1:
            success = search_time[-1, -1] < 10
        else:
            success = search_time[-1, -1] == 0
        progress.append(search_time[-1, -1])
        if success:
            if mcts_parameters.domain.find('convbelt') != -1:
                success_idx = np.where(search_time[:, -1] < 10)[0][0]
            else:
                success_idx = np.where(search_time[:, -1] == 0)[0][0]
            success_rewards.append(search_time[success_idx, 2])

        # print len(search_time), fin
        search_rwd_times.append(search_time)
        max_rwds.append(np.max(search_time[:, 2]))

    if not mcts_parameters.pw:
        print 'progress', np.array(progress).mean()
        print 'success reward', np.mean(success_rewards)

    print 'n_tested', len(progress)
    return search_rwd_times, np.mean(max_rwds)


def get_max_rwds_wrt_time(search_rwd_times):
    max_time = 15000
    n_evals = range(10, max_time, 1)

    all_episode_data = []
    all_episode_progress_data = []
    for rwd_time in search_rwd_times:
        episode_max_rwds_wrt_n_evals = []
        episode_max_progress_wrt_n_evals = []
        for organized_time in n_evals:
            episode_times = np.array(rwd_time)[:, 0]
            episode_rwds = np.array(rwd_time)[:, 2]
            idxs = episode_times < organized_time
            if np.any(idxs):
                max_rwd = np.max(episode_rwds[idxs])
            else:
                max_rwd = 0
            episode_progress = -np.array(rwd_time)[:, 3]
            # max_progress = np.max(episode_progress[idxs])
            episode_max_rwds_wrt_n_evals.append(max_rwd)
        # episode_max_progress_wrt_n_evals.append(max_progress)
        all_episode_data.append(episode_max_rwds_wrt_n_evals)
        all_episode_progress_data.append(episode_max_progress_wrt_n_evals)

    return np.array(all_episode_data), np.array(all_episode_progress_data), n_evals


def get_max_rwds_wrt_samples(search_rwd_times, n_evals):
    all_episode_data = []
    all_episode_progress_data = []
    for rwd_time in search_rwd_times:
        episode_max_rwds_wrt_n_evals = []
        episode_max_progress_wrt_n_evals = []
        for organized_time in range(n_evals):
            episode_times = np.array(rwd_time)[:, 1]
            episode_rwds = np.array(rwd_time)[:, 2]
            episode_progress = -np.array(rwd_time)[:, 3]
            idxs = episode_times <= organized_time
            max_rwd = np.max(episode_rwds[idxs])
            max_progress = np.max(episode_progress[idxs])
            episode_max_rwds_wrt_n_evals.append(max_rwd)
            episode_max_progress_wrt_n_evals.append(max_progress)
        all_episode_data.append(episode_max_rwds_wrt_n_evals)
        all_episode_progress_data.append(episode_max_progress_wrt_n_evals)
    return np.array(all_episode_data), np.array(all_episode_progress_data)


def get_algo_name_to_put_on_plot(raw_name):
    if raw_name.find('randomized_doo') != -1:
        return "RandDOOT"
    elif raw_name.find('voo') != -1:
        return 'VOOT'
    elif raw_name.find('unif') != -1:
        return "UniformT"
    elif raw_name.find('pw') != -1:
        return "PW-UCT"
    elif raw_name.find('doo') != -1:
        return "DOOT"
    elif 'RL_ppo' in raw_name:
        return "PPO"
    elif 'RL_ddpg' in raw_name:
        return "DDPG"
    else:
        raise ValueError


def setup_options():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-domain', type=str, default='minimum_displacement_removal_results')
    parser.add_argument('-w', type=float, default=5.0)
    parser.add_argument('-pw', type=float, default=0.1)
    parser.add_argument('-c1', type=int, default=1)
    parser.add_argument('-uct', type=float, default=0.0)
    parser.add_argument('-mcts_iter', type=int, default=2000)
    parser.add_argument('-n_feasibility_checks', type=int, default=50)
    parser.add_argument('-problem_idx', type=int, default=0)
    parser.add_argument('--p', action='store_true')
    parser.add_argument('-add', type=str, default='')
    parser.add_argument('-n_switch', type=int, default=10)
    parser.add_argument('-use_max_backup', action='store_true', default=False)
    parser.add_argument('-counter_ratio', type=int, default=1)
    parser.add_argument('-pick_switch', action='store_true', default=False)
    parser.add_argument('-n_actions_per_node', type=int, default=1)
    parser.add_argument('-value_threshold', type=float, default=-40.0)

    args = parser.parse_args()
    if args.domain == 'convbelt':
        domain = 'cbelt'
        args.mcts_iter = 3000
        args.voo_sampling_mode = 'uniform'
        args.n_switch = 10
        args.pick_switch = False
        args.use_max_backup = True
        args.n_feasibility_checks = 50
        args.problem_idx = 3
        args.w = 5.0
        args.n_actions_per_node = 3
        args.add = 'no_averaging'
    elif args.domain == 'mdr':
        domain = 'mdr'
        args.mcts_iter = 2000
        args.voo_sampling_mode = 'uniform'
        args.n_switch = 10
        args.pick_switch = True
        args.use_max_backup = True
        args.n_feasibility_checks = 50
        args.problem_idx = 0
        args.w = 5.0
        args.n_actions_per_node = 1
        args.add = 'no_averaging'
    elif args.domain.find('synthetic') != -1:
        domain = args.domain
        # note: I think for all objective functions, problem idx is 2?
        if args.problem_idx == 0:
            args.mcts_iter = 10000
            args.n_switch = 5
        elif args.problem_idx == 1:
            args.mcts_iter = 10000
            args.n_switch = 5
        elif args.problem_idx == 2:
            args.mcts_iter = 10000
            args.n_switch = 3
        else:
            raise NotImplementedError
        args.voo_sampling_mode = 'centered_uniform'
        args.pick_switch = False
        args.use_max_backup = True
        args.w = 100  # note: I think for griewank widening should be 100
        args.n_actions_per_node = 1
    else:
        raise NotImplementedError
    return args


def get_algo_names(options):
    if options.domain == 'convbelt':
        algo_names = ['RL_ppo', 'RL_ddpg', 'voo_uniform', 'pw', 'randomized_doo']
    elif options.domain == 'mdr':
        algo_names = ['RL_ppo', 'RL_ddpg', 'pw', 'voo_uniform', 'randomized_doo' ]
    elif options.domain.find('synthetic') != -1:
        if options.domain.find('shekel') != -1:
            algo_names = ['pw', 'voo_centered_uniform', 'doo']
        elif options.domain.find('griewank') != -1:
            algo_names = ['pw', 'voo_centered_uniform', 'doo']
        elif options.domain.find('rastrigin') != -1:
            algo_names = ['pw', 'voo_centered_uniform', 'doo']
    return algo_names


def get_color_for_each_algorithm():
    color_dict = pickle.load(open('./plotters/color_dict.p', 'r'))
    color_names = color_dict.keys()
    color_dict[color_names[0]] = [0., 0.5570478679, 0.]
    color_dict['RandDOOT'] = [0, 0, 1]
    color_dict['VOOT'] = [1, 0, 0]
    color_dict['PW-UCT'] = [0.8901960784313725, 0.6045098039215687, 0]
    color_dict['PPO'] = [98/255., 37/255., 147/255.]
    color_dict['DDPG'] = [15/255., 121/255., 147/255.]
    return color_dict


def get_color_for_algo(algo_name, color_dict):
    if algo_name in color_dict.keys():
        color = color_dict[algo_name]
    else:
        color = np.random.random((1, 3))

    return color


def get_plot_name(options):
    if options.p:
        plot_name = 'progress_toy_' + options.domain + '_problem_idx_' + str(options.problem_idx) + '_w_' + str(
            options.w) + '_mcts_iter_' + str(options.mcts_iter) \
                    + "_uct_" + str(options.uct) + "_n_feasibility_checks_" + str(
            options.n_feasibility_checks) + "_pw_" + str(options.pw)
        plot_name = options.domain+ '_progress'
    else:
        plot_name = 'reward_toy_' + options.domain + '_problem_idx_' + str(options.problem_idx) + '_w_' + str(
            options.w) \
                    + '_mcts_iter_' \
                    + str(options.mcts_iter) + "_uct_" + str(options.uct) + "_n_feasibility_checks_" \
                    + str(options.n_feasibility_checks) + '_use_max_backup_' + str(options.use_max_backup) \
                    + '_pick_switch_' + str(options.pick_switch) + "_pw_" + str(options.pw)

        if options.domain.find('synthetic') != -1:
            plot_name += '_value_threshold_' + str(options.value_threshold)

        if options.domain == 'convbelt_results':
            plot_name += '_n_actions_per_node_' + str(options.n_actions_per_node)

        if options.n_switch != -1:
            plot_name += "_n_switch_" + str(options.n_switch)
        plot_name = options.domain+ '_rewards'
    return plot_name


def plot_optimums(options):
    n_evals = options.mcts_iter
    if options.domain == 'mdr':
        if options.p:
            sns.tsplot([0] * n_evals, range(options.mcts_iter),
                       ci=95, condition='No objects remaining', color='magenta')
        else:
            sns.tsplot([4.1] * n_evals, range(options.mcts_iter), ci=95,
                       condition='Avg feasible reward', color='magenta')


def plot_across_algorithms():
    options = setup_options()
    algo_names = get_algo_names(options)
    color_dict = get_color_for_each_algorithm()

    for algo_idx, algo in enumerate(algo_names):
        print algo

        algo_name = get_algo_name_to_put_on_plot(algo)
        if 'PPO' in algo_name or 'DDPG' in algo_name:
            fdir = 'RL_results/%s/n_data_100/%s/dg_lr_0.001_0.0001/' % (options.domain, algo_name.lower())
            search_rwd, search_progress = plot_rl.get_max_rwds_wrt_samples(fdir)
        else:
            search_rwd_times, max_rwd = get_mcts_results(algo, options)
            search_rwd, search_progress = get_max_rwds_wrt_samples(search_rwd_times, options.mcts_iter)
        color = get_color_for_algo(algo_name, color_dict)

        content_to_plot = search_progress if options.p else search_rwd
        sns.tsplot(content_to_plot[:, :options.mcts_iter], ci=95, condition=algo_name, color=color)

    plot_name = get_plot_name(options)
    plot_optimums(options)

    if options.domain == 'cbelt':
        if not options.p:
            plt.ylim(-2, 4.5)
    y_axis_label = 'Number of remaining objects' if options.p else 'Average rewards'

    savefig('Number of simulations', y_axis_label, fname='./plotters/' + options.add + plot_name)


if __name__ == '__main__':
    plot_across_algorithms()
