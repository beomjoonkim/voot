import pickle
import argparse
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# todo I forgot to switch the plots in the submission, and accidentally used the one that takes the max and average them.


def savefig(xlabel, ylabel, fname=''):
    plt.legend(loc='best', prop={'size': 13})
    #plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    #plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    print 'Saving figure ', fname + '.png'
    plt.savefig(fname + '.png', dpi=100, format='png')


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

def get_result_dir(algo_name, dimension, obj_fcn):
    ROOTDIR = '/home/beomjoon/Dropbox (MIT)/braincloud/gtamp_results/'
    result_dir = ROOTDIR+'/function_optimization/' + str(obj_fcn) + '/dim_' + str(
        dimension) + '/' + algo_name + '/'
    return result_dir


def get_results(algo_name, dimension, obj_fcn):
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
        if algo_name == 'doo':
            optimal_epsilon_idx = np.argmax(max_ys[:, -1])
        else:
            optimal_epsilon_idx = np.argmax(max_ys[:, -1])
        max_y = max_ys[optimal_epsilon_idx, :]
        if len(max_y) < 500:
            continue
        else:
            max_y_values.append(max_y)
        print fin, len(max_y_values[-1]), max_y[-1], optimal_epsilon_idx

    print 'number of functions tested ', len(max_y_values)
    return np.array(max_y_values)


def plot_across_algorithms():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-dim', type=int, default=3)
    parser.add_argument('-obj_fcn', type=str, default='shekel')
    args = parser.parse_args()
    n_dim = args.dim

    algo_names = ['cmaes', 'rembo_ei', 'bamsoo', 'gpucb', 'soo', 'voo', 'doo', ]

    color_dict = {}
    color_dict['rembo_ei'] = [0., 0.5570478679, 0.]
    color_dict['cmaes'] = [0, 0, 0]
    color_dict['voo'] = [1, 0, 0]
    color_dict['doo'] = [0, 0, 1]
    color_dict['soo'] = [3 / 255.0, 252 / 255.0, 148 / 255.0]
    color_dict['bamsoo'] = [117 / 255.0, 15 / 255.0, 138 / 255.0]
    color_dict['gpucb'] = [15 / 255.0, 117 / 255.0, 138 / 255.0]

    ga_color = [0.2, 0.9, 0.1]
    ga_color = 'magenta'
    if args.obj_fcn != 'shekel':
        sns.tsplot([0]*2000, range(2000), ci=95, condition='Optimum', color='magenta')
    else:
        if n_dim == 3:
            plt.plot(range(2000), [4.89829] * 2000, linestyle='--', color=ga_color, label='GA_85k_evals')
        elif n_dim == 10:
            plt.plot(range(5000), [8.96] * 5000, linestyle='--', color=ga_color, label='GA_473k_evals')
        elif n_dim == 20:
            plt.plot(range(5000), [5.32] * 5000, linestyle='--', color=ga_color, label='GA_750k_evals')

    """
    if args.obj_fcn == 'rastrigin':
        if n_dim == 10:
            plt.plot(range(5000), [-21] * 5000, linestyle='--', color=ga_color, label='GA_40k_evals')
        elif n_dim == 20:
            plt.plot(range(5000), [-47] * 5000, linestyle='--', color=ga_color, label='GA_100k_evals')
    """

    if args.dim == 3 or args.obj_fcn == 'griewank':
        n_samples = 500
    elif args.obj_fcn == 'rosenbrock':
        n_samples = 5000
    elif args.obj_fcn == 'shekel' and args.dim == 20:
        n_samples = 5000
    else:
        n_samples = 1000

    for algo_idx, algo in enumerate(algo_names):
        print algo
        search_rwd_times = get_results(algo, n_dim, args.obj_fcn)
        if search_rwd_times is None:
            continue

        search_rwd_times = search_rwd_times[:, 0:n_samples]
        n_samples_tested = search_rwd_times.shape[-1]
        if n_samples_tested < n_samples:
            sns.tsplot(search_rwd_times, range(n_samples_tested), ci=95, condition=algo.upper(), color=color_dict[color_names[algo_idx]])
        else:
            sns.tsplot(search_rwd_times, range(n_samples), ci=95, condition=algo.upper(), color=color_dict[color_names[algo_idx]])
        print algo, n_samples, np.mean(search_rwd_times[:, -1])
        print "===================="

    plt.xlim(0, n_samples)
    savefig('Number of function evaluations', 'Best function values',
            fname='./plotters/' + args.obj_fcn + '_fcn_optimization_' + str(args.dim))


if __name__ == '__main__':
    plot_across_algorithms()
