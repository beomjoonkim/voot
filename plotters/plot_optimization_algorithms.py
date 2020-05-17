import pickle
import argparse
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def savefig(xlabel, ylabel, fname=''):
    plt.legend(loc='best', prop={'size': 13})
    # plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    # plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # print 'Saving figure ', fname + '.png'
    plt.show()
    import pdb;pdb.set_trace()
    plt.savefig(fname + '.png', dpi=100, format='png')


def get_result_dir(algo_name, dimension, obj_fcn):
    ROOTDIR = '/home/beomjoon/Dropbox (MIT)/braincloud/gtamp_results/'
    result_dir = ROOTDIR + '/function_optimization/' + str(obj_fcn) + '/dim_' + str(
        dimension) + '/' + algo_name + '/'
    return result_dir


def augment_cmaes_data(max_y, population_size=5):
    desired_length = (len(max_y)) * population_size
    augmented = []

    max_y_idx = 0
    for idx in range(desired_length):
        if idx % population_size == 0:
            curr_max_y = max_y[max_y_idx]
            max_y_idx += 1
        augmented.append(curr_max_y)
    return np.array(augmented)


def get_optimal_epsilon_idx(result_dir):
    eps_to_max_vals = {}
    try:
        result_files = os.listdir(result_dir)
    except OSError:
        return None

    for fin in result_files:
        if fin.find('.pkl') == -1:
            continue
        result = pickle.load(open(result_dir + fin, 'r'))
        max_ys = np.array(result['max_ys'])
        max_y = max_ys[0, :]
        if 'epsilon' not in result and 'epsilons' not in result:
            continue
        try:
            epsilons = result['epsilons']
        except KeyError:
            epsilons = result['epsilon']
        # print result['epsilons']
        print epsilons
        for idx, epsilon in enumerate(epsilons):
            if epsilon in eps_to_max_vals:
                eps_to_max_vals[epsilon].append(max_ys[idx, -1])
            else:
                eps_to_max_vals[epsilon] = [max_ys[idx, -1]]

    max_val = -np.inf
    for eps, val in zip(eps_to_max_vals.keys(), eps_to_max_vals.values()):
        eps_to_max_vals[eps] = np.mean(val)
        if np.mean(val) > max_val:
            max_val = np.mean(val)
            max_esp = eps
    print result_dir

    if 'rembo' in result_dir:
        return -1 # 0, 1, 2
    else:
        if max_esp in epsilons:
            return epsilons.index(max_esp)
        else:
            return epsilons[0]

def get_results_for_rastrigin(algo_name, dimension, obj_fcn):
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
        optimal_epsilon_idx = np.argmax(max_ys[:, -1])
        if 'rembo' in result_dir:
            optimal_epsilon_idx = -1
        if len(max_ys.shape) == 1:
            max_y = max_ys
        else:
            max_y = max_ys[optimal_epsilon_idx]

        ####
        if 'cmaes' in result_dir:
            if 'shekel' in result_dir and 'dim_20' in result_dir:
                max_y = augment_cmaes_data(max_y, 6)
            else:
                max_y = augment_cmaes_data(max_y, 5)

        print len(max_y), result_dir+fin
        if ('griewank' in obj_fcn or 'dim_3' in result_dir) and len(max_y) < 500:
                print "Skipping because not enough max_y", result_dir+fin
                continue
        elif 'shekel' in result_dir and 'dim_20' in result_dir:
            if len(max_y) < 5000:
                if 'bamsoo' in result_dir or 'rembo_ei' in result_dir or 'gpucb' in result_dir:
                    max_y = np.hstack([max_y, [max_y[-1]]*(5000-len(max_y))])
                else:
                    print "Skipping because not enough max_y", result_dir+fin
                    continue
        elif ('dim_3' not in  result_dir) and len(max_y) < 1000:
            print "Skipping because not enough max_y", result_dir+fin
            continue
        ###

        max_y_values.append(max_y)
        if len(max_y) < 500 and not ('cmaes' in result_dir):
            continue
        else:
            max_y_values.append(max_y)
        ####


        print fin, len(max_y_values[-1]), max_y[-1], optimal_epsilon_idx

    print 'number of functions tested ', len(max_y_values)
    return np.array(max_y_values)


def get_results(algo_name, dimension, obj_fcn):
    result_dir = get_result_dir(algo_name, dimension, obj_fcn)
    max_y_values = []

    try:
        result_files = os.listdir(result_dir)
    except OSError:
        print 'Exiting' + result_dir
        return None

    optimal_epsilon_idx = get_optimal_epsilon_idx(result_dir)
    for fin in result_files:
        if fin.find('.pkl') == -1:
            continue
        result = pickle.load(open(result_dir + fin, 'r'))
        max_ys = np.array(result['max_ys']).squeeze()

        if len(max_ys.shape) == 1:
            max_y = max_ys
        else:
            max_y = max_ys[optimal_epsilon_idx]

        if 'cmaes' in result_dir:
            if 'shekel' in result_dir and 'dim_20' in result_dir:
                max_y = augment_cmaes_data(max_y, 6)
            else:
                max_y = augment_cmaes_data(max_y, 5)

        print len(max_y), result_dir+fin

        if 'griewank' in obj_fcn or 'dim_3' in result_dir:
            if len(max_y) < 500:
                print "Skipping because not enough max_y", result_dir+fin
                continue
        elif 'shekel' in result_dir and 'dim_20' in result_dir:
            if len(max_y) < 5000:
                if 'bamsoo' in result_dir or 'rembo_ei' in result_dir or 'gpucb' in result_dir:
                    max_y = np.hstack([max_y, [max_y[-1]]*(5000-len(max_y))])
                else:
                    print "Skipping because not enough max_y", result_dir+fin
                    continue
        elif len(max_y) < 1000:
            print "Skipping because not enough max_y", result_dir+fin
            continue

        max_y_values.append(max_y)
        if len(max_y) < 500 and not ('cmaes' in result_dir):
            continue
        else:
            max_y_values.append(max_y)

    return np.array(max_y_values)


def plot_across_algorithms():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-dim', type=int, default=10)
    parser.add_argument('-obj_fcn', type=str, default='shekel')
    args = parser.parse_args()
    n_dim = args.dim
    if args.dim<10:
        algo_names = ['cmaes', 'bamsoo', 'gpucb', 'soo', 'voo', 'doo'] #, 'uniform']
    else:
        algo_names = ['cmaes', 'rembo_ei', 'bamsoo', 'gpucb', 'soo', 'voo', 'doo'] #, 'uniform']
        algo_names = ['rembo_ei', 'bamsoo', 'gpucb', 'soo', 'doo', 'cmaes', 'voo']
    #algo_names = ['rembo_ei', 'bamsoo', 'gpucb', 'soo', 'voo', 'doo', ]

    color_dict = {}
    color_dict['rembo_ei'] = [0., 0.5570478679, 0.]
    color_dict['cmaes'] = [0, 0, 0]
    color_dict['voo'] = [1, 0, 0]
    color_dict['doo'] = [0, 0, 1]
    color_dict['soo'] = [3 / 255.0, 252 / 255.0, 148 / 255.0]
    color_dict['bamsoo'] = [117 / 255.0, 15 / 255.0, 138 / 255.0]
    color_dict['gpucb'] = [139 / 255.0, 69 / 255.0, 19 / 255.0]
    color_dict['uniform'] = [139 / 255.0, 69 / 255.0, 100/ 255.0]
    optimum_color = 'magenta'
    if args.obj_fcn != 'shekel':
        sns.tsplot([0] * 2000, range(2000), ci=95, condition='Optimum', color='magenta')
    else:
        if n_dim == 3:
            plt.plot(range(2000), [4.78739] * 2000, linestyle='--', color=optimum_color, label='GA_2.4e4_evals')
        elif n_dim == 10:
            plt.plot(range(5000), [6.04759] * 5000, linestyle='--', color=optimum_color, label='GA_2.65e5_evals')
        elif n_dim == 20:
            plt.plot(range(5000), [3.93869] * 5000, linestyle='--', color=optimum_color, label='GA_8.10e5_evals')

    if args.dim == 3 or args.obj_fcn == 'griewank':
        n_samples = 500
    elif args.obj_fcn == 'rosenbrock':
        n_samples = 5000
    elif args.obj_fcn == 'shekel' and args.dim == 20:
        n_samples = 5000
    else:
        n_samples = 1000

    for algo_idx, algo in enumerate(algo_names):
        # print algo
        if 'rastrigin' == args.obj_fcn:
            search_rwd_times = get_results_for_rastrigin(algo, n_dim, args.obj_fcn)
        else:
            search_rwd_times = get_results(algo, n_dim, args.obj_fcn)
        if search_rwd_times is None:
            continue

        search_rwd_times = search_rwd_times[:, 0:n_samples]
        #search_rwd_times = search_rwd_times[np.argsort(search_rwd_times[:, -1])[10:], :]
        n_samples_tested = search_rwd_times.shape[-1]
        if algo == 'rembo_ei':
            algo_name = 'REMBO'
        elif algo == 'bamsoo':
            algo_name = 'BaMSOO'
        else:
            algo_name = algo.upper()

        if n_samples_tested < n_samples:
            sns.tsplot(search_rwd_times, range(n_samples_tested), ci=95, condition=algo_name,
                       color=color_dict[algo])
        else:
            sns.tsplot(search_rwd_times, range(n_samples), ci=95, condition=algo_name,
                       color=color_dict[algo])
        # print algo, n_samples, np.mean(search_rwd_times[:, -1])
        # print "===================="

    plt.xlim(-20, n_samples)
    savefig('Number of function evaluations', 'Best function values',
            fname='./plotters/' + args.obj_fcn + '_fcn_optimization_' + str(args.dim))


if __name__ == '__main__':
    plot_across_algorithms()
