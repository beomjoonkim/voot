import numpy as np
from doo import DOOTreeNode, BinaryDOOTree


def deterministic_optimistic_optimization(f, n_evals=50):
    evaled_xs = []
    evaled_scores = []

    x = 0.5
    evaled_xs.append(x)
    evaled_scores.append(f(x))

    # cut the plane in half
    left_bound = 0
    right_bound = 1

    root_node = DOOTreeNode(x, f(x), min_x=left_bound, max_x=right_bound)
    tree = BinaryDOOTree(root_node)

    while len(evaled_xs) < n_evals:
        next_node = tree.choose_leaf_node_to_expand()
        left_x, left_fx = tree.add_left(f, next_node)
        right_x, right_fx = tree.add_right(f, next_node)
        evaled_xs.append(left_x)
        evaled_xs.append(right_x)
        evaled_scores.append(left_fx)
        evaled_scores.append(right_fx)

    return evaled_xs, evaled_scores


def make_objective_function():
    th1 = np.random.random_sample((1,)) * (1 - 0.5) + 0.5
    th2 = np.random.random_sample((1,)) * (20 - 1) + 1
    th3 = np.random.random_sample((1,)) * (30 - 20) + 20
    f = lambda x: th1 * (np.sin(th2 * x) * np.sin(th3 * x)) + 0.5
    almost_all_x = np.random.random_sample((10000,)) * (1 - -0) + (0)
    f_vals = f(almost_all_x)
    f_star = max(f_vals)

    return f, f_star

def test_doo(n_evals):
    regrets = []
    for i in range(100):
        f, f_star = make_objective_function()
        evaled_xs, evaled_fx_vals = deterministic_optimistic_optimization(f, n_evals)
        regrets.append(np.sort((f_star - evaled_fx_vals).squeeze())[::-1])

    visualize_evaluations(evaled_xs, f, 'DOO')
    return np.array(regrets)

