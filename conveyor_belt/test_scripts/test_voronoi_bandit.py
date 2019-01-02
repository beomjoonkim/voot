import matplotlib.pyplot as plt
from deterministic_optimistic_optimization import binary_DOO_tree, DOO_tree_node
from sampling_strategies.voo import AnalyticalVOO

import numpy as np
import time


def make_objective_function():
    th1 = np.random.random_sample((1,)) * (1 - 0.5) + 0.5
    th2 = np.random.random_sample((1,)) * (20 - 1) + 1
    th3 = np.random.random_sample((1,)) * (30 - 20) + 20
    f = lambda x: th1 * (np.sin(th2 * x) * np.sin(th3 * x)) + 0.5
    almost_all_x = np.random.random_sample((10000,)) * (1 - -0) + (0)
    f_vals = f(almost_all_x)
    f_star = max(f_vals)

    return f, f_star


def distance(x, y):
    return abs(x - y)


def visualize_evaluations(evaled_xs, f, algorithm_name):
    almost_all_x = np.random.random_sample((10000,)) * (1 - -0) + (0)
    plt.scatter(almost_all_x, f(almost_all_x), c='red', label='f', marker='o')
    plt.scatter(evaled_xs, f(evaled_xs), c='blue', label='evaluated', marker='o')
    plt.savefig('./test_scripts/test_results/'+algorithm_name+'.png')
    plt.close('all')


def sample_from_best_voroi_region(evaled_xs, evaled_scores):
    best_x = evaled_xs[np.argmax(evaled_scores)]
    x = np.random.random()

    dists = np.array([distance(x, y) for y in evaled_xs if y != best_x])
    best_x_dist = np.array(distance(x, best_x))
    while any(best_x_dist > dists):
        x = np.random.random()
        best_x_dist = np.array(distance(x, best_x))

    return x




def deterministic_optimistic_optimization(f, n_evals=50):
    evaled_xs = []
    evaled_scores = []

    x = 0.5
    evaled_xs.append(x)
    evaled_scores.append(f(x))

    # cut the plane in half
    left_bound = 0
    right_bound = 1

    root_node = DOO_tree_node(x, f(x), min_x=left_bound, max_x=right_bound)
    tree = binary_DOO_tree(root_node)

    #visualize_evaluations(evaled_xs, f, 'DOO'+str(len(evaled_xs)))
    while len(evaled_xs) < n_evals:
        next_node = tree.choose_leaf_node_to_expand()
        left_x, left_fx = tree.add_left(f, next_node)
        right_x, right_fx = tree.add_right(f, next_node)
        evaled_xs.append(left_x)
        evaled_xs.append(right_x)
        evaled_scores.append(left_fx)
        evaled_scores.append(right_fx)
        #visualize_evaluations([leaf.x_value for leaf in tree.leaves], f, 'DOO'+str(len(evaled_xs)))

    return evaled_xs, evaled_scores


def test_doo(n_evals):
    regrets = []
    for i in range(100):
        f, f_star = make_objective_function()
        evaled_xs, evaled_fx_vals = deterministic_optimistic_optimization(f, n_evals)
        regrets.append(np.sort((f_star - evaled_fx_vals).squeeze())[::-1])

    visualize_evaluations(evaled_xs, f, 'DOO')
    return np.array(regrets)


def test_uniform(n_evals):
    regrets = []
    n_evals += 1
    for i in range(100):
        f, f_star = make_objective_function()
        x_vals = np.random.random_sample((n_evals,)) * (1 - -0) + (0)
        f_vals = f(x_vals)

        regrets.append(np.sort((f_star - f_vals).squeeze())[::-1])
    return np.array(regrets)


def voronoi_optimistic_optimization(explr_p, f, n_evals=50):
    evaled_xs = []
    evaled_scores = []

    x = np.random.random_sample((1,)) * (1 - -0) + (0)
    evaled_xs.append(x[0])
    evaled_scores.append(f(x))

    for i in range(n_evals):
        if np.random.random() < 1-explr_p:
            x = sample_from_best_voroi_region(evaled_xs, evaled_scores)
        else:
            x = np.random.random()
        evaled_xs.append(x)
        evaled_scores.append(f(x))

    return evaled_xs, evaled_scores


def analytical_voronoi_optimistic_optimization(explr_p, f, n_evals=50):
    evaled_xs = []
    evaled_scores = []

    x = np.random.random_sample((1,)) * (1 - -0) + (0)
    evaled_xs.append(x[0])
    evaled_scores.append(f(x))

    for i in range(n_evals):
        if np.random.random() < 1-explr_p:
            x = sample_from_best_voroi_region(evaled_xs, evaled_scores)
        else:
            x = np.random.random()
        evaled_xs.append(x)
        evaled_scores.append(f(x))

    return evaled_xs, evaled_scores


def test_voroi_bandit(n_evals, explr_p):
    regrets = []
    for i in range(100):
        f, f_star = make_objective_function()
        evaled_xs, evaled_fx_vals = voronoi_optimistic_optimization(explr_p, f, n_evals=n_evals)
        regrets.append(np.sort((f_star - evaled_fx_vals).squeeze())[::-1])

    visualize_evaluations(evaled_xs, f, 'DOO')
    return np.array(regrets)


def test_analytical_voroi_bandit(n_evals, explr_p):
    regrets = []
    for i in range(100):
        f, f_star = make_objective_function()
        evaled_xs, evaled_fx_vals = analytical_voronoi_optimistic_optimization(explr_p, f, n_evals=n_evals)
        regrets.append(np.sort((f_star - evaled_fx_vals).squeeze())[::-1])

    visualize_evaluations(evaled_xs, f, 'DOO')
    return np.array(regrets)


def main():
    analytical_voo = test_analytical_voroi_bandit(n_evals=100)  # todo  implement this function
    import pdb;pdb.set_trace()
    voo_regrets_zero_eps = test_voroi_bandit(n_evals=100, explr_p=0)
    voo_regrets_nonzero_eps = test_voroi_bandit(n_evals=100, explr_p=0.9)
    doo_regrets = test_doo(n_evals=100)
    unif_regrets = test_uniform(n_evals=100)

    plt.figure()
    plt.errorbar(range(1, 102), voo_regrets_zero_eps.mean(axis=0), 0 * voo_regrets_zero_eps.std(axis=0), label='VOO_zero')
    plt.errorbar(range(1, 102), voo_regrets_nonzero_eps.mean(axis=0), 0 * voo_regrets_nonzero_eps.std(axis=0), label='VOO_eps_0.3')
    plt.errorbar(range(1, doo_regrets.shape[1]+1), doo_regrets.mean(axis=0), 0 * doo_regrets.std(axis=0), label='DOO')
    plt.errorbar(range(1, 102), unif_regrets.mean(axis=0), 0 * unif_regrets.std(axis=0), label='Uniform')
    plt.legend()
    plt.show()
    import pdb;
    pdb.set_trace()


if __name__ == '__main__':
    main()
