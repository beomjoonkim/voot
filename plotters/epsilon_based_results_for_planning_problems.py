from plot_planning import *


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
    return result_dir


def print_hyperparameter_results(result_dir, problem_idx):
    fdirs = glob.glob(result_dir)
    print 'n hyper-parameters tested', len(fdirs)
    if len(fdirs) == 0:
        print result_dir + " is empty"
        sys.exit(-1)
    for fidx, fdir in enumerate(fdirs):
        last_rwds = []
        half_time_rwds = []
        first_quarter_rwds = []
        listfiles = os.listdir(fdir)
        is_pw = fdir.find('pw_methods') != -1
        uct = float(fdir.split('uct_')[1].split('_')[0])
        widening = float(fdir.split('widening_')[1].split('_')[0])

        if len(listfiles) < 10:
            continue
        for fin in listfiles:
            if fin.find('pidx') == -1:
                continue
            file_problem_idx = int(fin.split('_')[-1].split('.')[0])
            if file_problem_idx != problem_idx:
                continue

            result = pickle.load(open(fdir + fin, 'r'))
            search_time = np.array(result['search_time'])
            max_rwd = search_time[-1, -2]
            half_time = search_time.shape[0] / 2
            half_time_rwd = search_time[half_time, -2]

            first_quarter = search_time.shape[0] / 4
            first_quarter_rwd = search_time[first_quarter, -2]

            last_rwds.append(max_rwd)
            half_time_rwds.append(half_time_rwd)
            first_quarter_rwds.append(first_quarter_rwd)

        if result_dir.find('synthetic') == -1:
            results = " $%.4f\pm%.4f$" % (
                np.mean(first_quarter_rwds), 1.96 * np.std(first_quarter_rwds) / np.sqrt(len(first_quarter_rwds)))
            results += " $%.5f\pm%.4f$" % (
                np.mean(half_time_rwds), 1.96 * np.std(half_time_rwds) / np.sqrt(len(half_time_rwds)))
            results += " $%.4f\pm%.4f$" % (np.mean(last_rwds), 1.96 * np.std(last_rwds) / np.sqrt(len(last_rwds)))
        else:
            results = " $%.4f\pm%.4f$" % (np.mean(last_rwds), 1.96 * np.std(last_rwds) / np.sqrt(len(last_rwds)))
        if is_pw:
            config = "%.2f %.2f" % (uct, widening)
        else:
            epsilon = float(fdir.split('eps_')[1].split('/')[0])
            config = str(epsilon) #"%.3f" % (epsilon)
        print config + results
    print "Niter = ", len(search_time)


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
    parser.add_argument('-value_threshold', type=float, default=-40.0)

    args = parser.parse_args()
    if args.domain.find('results') == -1:
        args.domain += '_results'

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
        args.voo_sampling_mode = 'uniform'
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
        args.voo_sampling_mode = 'uniform'
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
        if args.domain.find('griewank') == -1:
            args.w = 100
        else:
            args.w = 5.0
        args.n_actions_per_node = 1
    else:
        raise NotImplementedError

    if args.domain == 'convbelt_results':
        algo_names = ['pw', 'randomized_doo', 'voo_uniform']
    elif args.domain == 'minimum_displacement_removal_results':
        algo_names = ['pw', 'voo_uniform', 'randomized_doo']
    elif args.domain.find('synthetic') != -1:
        algo_names = ['pw', 'voo_centered_uniform', 'doo']

    for algo_idx, algo in enumerate(algo_names):
        print "Algo name " + algo
        result_dir = get_result_dir(algo, args)
        print_hyperparameter_results(result_dir, args.problem_idx)


if __name__ == '__main__':
    plot_across_algorithms()
