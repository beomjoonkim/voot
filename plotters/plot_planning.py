import cPickle as pickle
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import glob


def savefig(xlabel, ylabel, fname=''):
    plt.legend(loc='best', prop={'size': 13})
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    print 'Saving figure ', fname + '.png'
    plt.savefig(fname + '.png', dpi=100, format='png')


def get_best_hyperparameter_dir(result_dir, problem_idx):
    best_dir = None
    best_rwd_among_all_setups = -np.inf
    fdirs = glob.glob(result_dir)

    for fidx, fdir in enumerate(fdirs):
        #print "Going through %d / %d" % (fidx, len(fdirs))
        last_rwds = []
        listfiles = os.listdir(fdir)
        if len(listfiles) < 10:
            continue
        for fin in listfiles:
            if fin.find('pidx') == -1:
                #print "Skipping file", fin
                #print "Continuing"
                continue

            sd = int(fin.split('_')[2])
            file_problem_idx = int(fin.split('_')[-1].split('.')[0])

            if file_problem_idx != problem_idx:
                #print "Skipping file", fin
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

    rootdir = './test_results/'
    result_dir = rootdir + '/' + mcts_parameters.domain + '/mcts_iter_' + str(mcts_iter)
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

    if addendum != '':
        if algo_name != 'pw':
            result_dir += '_' + addendum + '/'
        else:
            result_dir += '_pw_reevaluates_infeasible/'
    else:
        result_dir += '/'

    problem_idx = mcts_parameters.problem_idx

    if algo_name == 'voo':
        result_dir += '/sampling_mode/'+voo_sampling_mode+'/counter_ratio_1/eps_*/'
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
            #print "Continuing"
            continue
        sd = int(fin.split('_')[2])
        file_problem_idx = int(fin.split('_')[-1].split('.')[0])

        if file_problem_idx != problem_idx:
            #print "Continuing"
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
    organized_times = range(10, max_time, 1)

    all_episode_data = []
    all_episode_progress_data = []
    for rwd_time in search_rwd_times:
        episode_max_rwds_wrt_organized_times = []
        episode_max_progress_wrt_organized_times = []
        for organized_time in organized_times:
            episode_times = np.array(rwd_time)[:, 0]
            episode_rwds = np.array(rwd_time)[:, 2]
            idxs = episode_times < organized_time
            if np.any(idxs):
                max_rwd = np.max(episode_rwds[idxs])
            else:
                max_rwd = 0
            episode_progress = -np.array(rwd_time)[:, 3]
            # max_progress = np.max(episode_progress[idxs])
            episode_max_rwds_wrt_organized_times.append(max_rwd)
        # episode_max_progress_wrt_organized_times.append(max_progress)
        all_episode_data.append(episode_max_rwds_wrt_organized_times)
        all_episode_progress_data.append(episode_max_progress_wrt_organized_times)

    return np.array(all_episode_data), np.array(all_episode_progress_data), organized_times


def get_max_rwds_wrt_samples(search_rwd_times, n_evals):
    organized_times = range(n_evals)

    all_episode_data = []
    all_episode_progress_data = []
    for rwd_time in search_rwd_times:
        episode_max_rwds_wrt_organized_times = []
        episode_max_progress_wrt_organized_times = []
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
    else:
        raise ValueError


def plot_across_algorithms():
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

    args = parser.parse_args()

    if args.domain == 'convbelt_results':
        domain_name = 'cbelt'
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
    elif args.domain == 'minimum_displacement_removal_results':
        domain_name = 'mdr'
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
        domain_name = args.domain
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
        args.w = 5.0
        args.n_actions_per_node = 1
    else:
        raise NotImplementedError

    if args.domain == 'convbelt_results':
        algo_names = ['pw', 'randomized_doo', 'voo_uniform']
    elif args.domain == 'minimum_displacement_removal_results':
        algo_names = ['pw', 'voo_uniform', 'randomized_doo']
    elif args.domain.find('synthetic') != -1:
        if args.domain.find('shekel') != -1:
            algo_names = ['voo_centered_uniform', 'doo']
        elif args.domain.find('griewank') != -1:
            algo_names = ['pw', 'voo_centered_uniform', 'doo']
        elif args.domain.find('rastrigin') != -1:
            algo_names = ['voo_centered_uniform', 'doo']
            algo_names = ['pw', 'voo_centered_uniform', 'doo']
    algo_names = ['voo_centered_uniform', 'doo']

    color_dict = pickle.load(open('./plotters/color_dict.p', 'r'))
    color_names = color_dict.keys()
    color_dict[color_names[0]] = [0., 0.5570478679, 0.]
    color_dict['RandDOOT'] = [0, 0, 1]
    color_dict['DOOT'] = [0, 0, 1]
    color_dict['VOOT'] = [1, 0, 0]
    #color_dict['PW-UCT'] = np.array([0, 100, 0]) / 255.0
    color_dict['PW-UCT'] = [0.8901960784313725, 0.6745098039215687, 0]

    max_rwds = []
    for algo_idx, algo in enumerate(algo_names):
        print algo
        try:
            search_rwd_times, max_rwd = get_mcts_results(algo, args)
        except OSError:
            print algo, "not found"
            continue
        search_rwd, search_progress, organized_times = get_max_rwds_wrt_samples(search_rwd_times, args.mcts_iter)

        max_rwds.append(max_rwd)
        algo_name = get_algo_name(algo)

        if algo_name in color_dict.keys():
            color = color_dict[algo_name]
        else:
            color = np.random.random((1, 3))

        if args.p:
            sns.tsplot(search_progress[:, :args.mcts_iter], organized_times[:args.mcts_iter], ci=95,
                       condition=algo_name,
                       color=color)
        else:
            try:
                sns.tsplot(search_rwd[:, :], organized_times[:], ci=95, condition=algo_name,
                           color=color)
            except:
                continue

    if args.p:
        plot_name = 'progress_toy_' + domain_name + '_problem_idx_' + str(args.problem_idx) + '_w_' + str(
            args.w) + '_mcts_iter_' + str(args.mcts_iter) \
                    + "_uct_" + str(args.uct) + "_n_feasibility_checks_" + str(
            args.n_feasibility_checks) + "_pw_" + str(args.pw)
        if args.domain != 'convbelt_results':
            sns.tsplot([0] * len(organized_times[:]), organized_times[:args.mcts_iter],
                       ci=95, condition='Avg feasible reward', color='magenta')
    else:
        if domain_name == 'mdr':
            sns.tsplot([4.1] * len(organized_times[:]), organized_times[:args.mcts_iter],
                       ci=95, condition='Avg feasible reward', color='magenta')

        plot_name = 'reward_toy_' + domain_name + '_problem_idx_' + str(args.problem_idx) + '_w_' + str(
            args.w) + '_mcts_iter_' \
                    + str(args.mcts_iter) + "_uct_" + str(args.uct) + "_n_feasibility_checks_" \
                    + str(args.n_feasibility_checks) + '_use_max_backup_' + str(args.use_max_backup) \
                    + '_pick_switch_' + str(args.pick_switch) + "_pw_" + str(args.pw)

        if args.domain == 'convbelt_results':
            plot_name += '_n_actions_per_node_' + str(args.n_actions_per_node)
        if args.n_switch != -1:
            plot_name += "_n_switch_" + str(args.n_switch)

    if domain_name == 'cbelt':
        if not args.p:
            plt.ylim(-2, 4.5)
    if args.p:
        savefig('Number of simulations', 'Number of remaining objects',
                fname='./plotters/' + args.add + '_toy_' + plot_name)
    else:
        savefig('Number of simulations', 'Average rewards', fname='./plotters/' + args.add + '_toy_' + plot_name)


if __name__ == '__main__':
    plot_across_algorithms()
