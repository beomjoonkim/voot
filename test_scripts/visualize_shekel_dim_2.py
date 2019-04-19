import pickle
import time
import os
import argparse
import numpy as np
import random

from deap.benchmarks import shekel
from deap import benchmarks

import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

from generators.voo_utils.voo import VOO
from generators.doo_utils.doo_tree import BinaryDOOTree
from generators.soo_utils.soo_tree import BinarySOOTree

parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('-algo_name', type=str, default='voo')
parser.add_argument('-n_fcn_evals', type=int, default=100)
parser.add_argument('-seed', type=int, default=1)
args = parser.parse_args()

algo_name = args.algo_name
dim_x = 2
n_fcn_evals = args.n_fcn_evals

# todo create multiple local optima
"""
config = pickle.load(open('./test_results/function_optimization/shekel/shekel_dim_'+str(2)+'.pkl', 'r'))
A = config['A']
C = config['C']
"""

NUMMAX=2

#A = np.random.rand(NUMMAX, dim_x)*10
#C = np.random.rand(NUMMAX)*10

A = np.array([[
    0.5, 0.5],
    [0.25, 0.25],
    [0.25, 0.75],
    [0.75, 0.25],
    [0.75, 0.75]
]) * 500
C = np.array([0.002, 0.005, 0.005, 0.005, 0.005]) * 500
domain = np.array([[-500.]*dim_x, [500.]*dim_x])
domain = np.array([[0.]*dim_x, [500.]*dim_x])

# todo define X near 0.5,0.5 , and 500,500. Both shows difference in mu(R)/mu(X)
save_dir = './test_results/function_optimization/visualization/shekel' + '/dim_' + str(dim_x) + \
           '/'+algo_name+'/'


def get_objective_function(sol):
    return shekel(sol, A, C)[0]


def random_search(epsilon):
    evaled_x = []
    evaled_y = []
    max_y = []
    dim_parameters = domain.shape[-1]
    domain_min = domain[0]
    domain_max = domain[1]
    times = []
    stime = time.time()
    for i in range(n_fcn_evals):
        #if i == 0:
        #    x = (domain_min+domain_max)/2.0
        #else:
        x = np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()
        if len(x.shape) == 0:
            x = np.array([x])
        y = get_objective_function(x)
        evaled_x.append(x)
        evaled_y.append(y)
        max_y.append(np.max(evaled_y))
        times.append(time.time()-stime)
    return evaled_x, evaled_y, max_y, times


def doo(explr_p, ax):
    distance_fn = lambda x, y: np.linalg.norm(x - y)
    doo_tree = BinaryDOOTree(domain, explr_p, distance_fn)

    evaled_x = []
    evaled_y = []
    max_y = []
    times = []
    stime = time.time()
    for i in range(n_fcn_evals):
        next_node = doo_tree.get_next_point_and_node_to_evaluate()
        x_to_evaluate = next_node.cell_mid_point
        next_node.evaluated_x = x_to_evaluate
        fval = get_objective_function(x_to_evaluate)
        doo_tree.expand_node(fval, next_node)

        evaled_x.append(x_to_evaluate)
        evaled_y.append(fval)
        max_y.append(np.max(evaled_y))
        times.append(time.time()-stime)

        draw_points(evaled_x, ax)
    print "Max value found", np.max(evaled_y)
    return evaled_x, evaled_y, max_y, times


def soo(dummy, ax):
    soo_tree = BinarySOOTree(domain)

    evaled_x = []
    evaled_y = []
    max_y = []
    times = []

    stime = time.time()
    for i in range(n_fcn_evals):
        next_node = soo_tree.get_next_point_and_node_to_evaluate()
        x_to_evaluate = next_node.cell_mid_point
        next_node.evaluated_x = x_to_evaluate
        fval = get_objective_function(x_to_evaluate)
        soo_tree.expand_node(fval, next_node)

        evaled_x.append(x_to_evaluate)
        evaled_y.append(fval)
        max_y.append(np.max(evaled_y))
        times.append(time.time()-stime)

        draw_points(evaled_x, ax)
    print "Max value found", np.max(evaled_y)
    return evaled_x, evaled_y, max_y, times


def voo(explr_p, ax):
    evaled_x = []
    evaled_y = []
    max_y = []
    voo = VOO(domain, explr_p)
    times = []
    stime = time.time()
    print 'explr_p', explr_p

    for i in range(n_fcn_evals):
        x = voo.choose_next_point(evaled_x, evaled_y)
        if len(x.shape) == 0:
            x = np.array([x])
        y = get_objective_function(x)
        evaled_x.append(x)
        evaled_y.append(y)
        max_y.append(np.max(evaled_y))
        times.append(time.time()-stime)

        draw_points(evaled_x, ax)

    best_idx = np.where(evaled_y == max_y[-1])[0][0]
    print evaled_x[best_idx], evaled_y[best_idx]
    print "Max value found", np.max(evaled_y)
    print "Magnitude", np.linalg.norm(evaled_x[best_idx])
    return evaled_x, evaled_y, max_y, times


def shekel_arg0(sol):
    return benchmarks.shekel(sol, A, C)[0]


def draw_shekel():
    from matplotlib import cm
    from matplotlib.colors import LogNorm

    X = np.arange(domain[0][0], domain[1][0], 5)
    Y = np.arange(domain[0][1], domain[1][1], 5)
    X, Y = np.meshgrid(X, Y)
    print "Evaluating shekel"
    Z = np.fromiter(map(shekel_arg0, zip(X.flat, Y.flat)), dtype=np.float, count=X.shape[0] * X.shape[1]).reshape(
        X.shape)
    ax = plt.axes(projection='3d')
    print "Drawing shekel"
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, norm=LogNorm(), cmap=cm.jet, linewidth=0.2)
    plt.savefig(save_dir+'/'+str(0)+'.png')
    print "Done!"
    return ax


def draw_points(points, ax):
    points = np.array(points)
    evaluations = [benchmarks.shekel(x, A, C)[0] for x in points]
    ax.scatter(points[:, 0], points[:, 1], evaluations, c='black')
    ax.scatter(points[-1, 0], points[-1, 1], evaluations[-1], c='red')
    plt.savefig(save_dir+'/'+str(points.shape[0])+'.png')


def main():
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if algo_name == 'uniform':
        algorithm = random_search
    elif algo_name == 'voo':
        algorithm = voo
    elif algo_name == 'doo':
        algorithm = doo
    elif algo_name == 'soo':
        algorithm = soo
    else:
        print "Wrong algo name"
        return

    if algo_name == 'voo':
        epsilon = 0.3
    elif algo_name == 'soo':
        epsilon = 0
    elif algo_name == 'doo':
        epsilon = 0.00001
    else:
        raise NotImplementedError

    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    ax = draw_shekel()
    evaled_x, evaled_y, max_y, time_taken = algorithm(epsilon, ax)




if __name__ == '__main__':
    main()


