from plot_optimization_algorithms import *


def get_results(algo_name, dimension, obj_fcn):
    result_dir = get_result_dir(algo_name, dimension, obj_fcn)
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
        if len(max_y) < 500:
            continue

        for idx, epsilon in enumerate(result['epsilons']):
            if epsilon in eps_to_max_vals.keys():
                eps_to_max_vals[epsilon].append(max_ys[idx, -1])
            else:
                eps_to_max_vals[epsilon] = [max_ys[idx, -1]]

    return eps_to_max_vals


def plot_across_algorithms():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-dim', type=int, default=3)
    parser.add_argument('-obj_fcn', type=str, default='shekel')
    parser.add_argument('-algo_name', type=str, default='gpucb')
    args = parser.parse_args()
    n_dim = args.dim

    algo_names = [args.algo_name]
    color_dict = pickle.load(open('./plotters/color_dict.p', 'r'))
    color_names = color_dict.keys()
    color_dict[color_names[0]] = [0., 0.5570478679, 0.]
    color_dict[color_names[1]] = [0, 0, 0]
    color_dict[color_names[2]] = [1, 0, 0]
    color_dict[color_names[3]] = [0, 0, 1]
    color_dict[color_names[4]] = [0.8901960784313725, 0.6745098039215687, 0]

    ga_color = [0.2, 0.9, 0.1]
    ga_color = 'magenta'

    if args.dim == 3 or args.obj_fcn == 'griewank':
        n_samples = 500
    elif args.obj_fcn == 'rosenbrock':
        n_samples = 5000
    elif args.obj_fcn == 'shekel' and args.dim == 20:
        n_samples = 5000
    else:
        n_samples = 1000

    for algo_idx, algo in enumerate(algo_names):
        print '====='+str(algo)+'===='
        eps_to_max_vals = get_results(algo, n_dim, args.obj_fcn)
        sorted_eps = np.sort(eps_to_max_vals.keys())
        for eps in sorted_eps:
            val = eps_to_max_vals[eps]
            print str(eps) + ' %.5f %.5f' % (np.mean(val), 1.96 * np.std(val)/np.sqrt(len(val)))
        print "===================="



if __name__ == '__main__':
    plot_across_algorithms()
