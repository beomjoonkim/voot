from deap.benchmarks import shekel
import matplotlib.pyplot as plt
import numpy as np

from generators.gpucb_utils.gp import StandardContinuousGP
from generators.gpucb_utils.functions import UCB, Domain
from generators.gpucb_utils.bo import BO
from generators.voo_utils.voo import VOO
from generators.doo_utils.doo_tree import BinaryDOOTree

import seaborn as sns
import pickle
import time


NUMMAX = 10
dim_x = 20
n_iter = 1000
domain = np.array([[0]*dim_x, [10]*dim_x])


def gpucb(explr_p):
    gp = StandardContinuousGP(dim_x)
    acq_fcn = UCB(zeta=explr_p, gp=gp)
    gp_format_domain = Domain(0, domain)
    gp_optimizer = BO(gp, acq_fcn, gp_format_domain)  # this depends on the problem

    evaled_x = []
    evaled_y = []
    max_y = []
    times = []
    stime = time.time()
    for i in range(n_iter):
        if i < 50:
            x = gp_optimizer.choose_next_point(evaled_x, evaled_y)
            y = shekel_arg0(x)
            evaled_x.append(x)
            evaled_y.append(y)
        else:
            evaled_x.append(x)
            evaled_y.append(y)
        max_y.append(np.max(evaled_y))
        times.append(time.time()-stime)

    return evaled_x, evaled_y, max_y, times


def voo(explr_p, obj_fcn):
    evaled_x = []
    evaled_y = []
    max_y = []
    voo = VOO(domain, explr_p)
    times = []
    stime = time.time()
    for i in range(n_iter):
        print i
        x = voo.choose_next_point(evaled_x, evaled_y)
        y = obj_fcn(x)
        evaled_x.append(x)
        evaled_y.append(y)
        max_y.append(np.max(evaled_y))
        times.append(time.time()-stime)
    return evaled_x, evaled_y, max_y, times


def random_search(epsilon, obj_fcn):
    evaled_x = []
    evaled_y = []
    max_y = []
    dim_parameters = domain.shape[-1]
    domain_min = domain[0]
    domain_max = domain[1]
    times = []
    stime = time.time()
    for i in range(n_iter):
        x= np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()
        y = obj_fcn(x)
        evaled_x.append(x)
        evaled_y.append(y)
        max_y.append(np.max(evaled_y))
        times.append(time.time()-stime)
    return evaled_x, evaled_y, max_y, times


def doo(explr_p,obj_fcn):
    distance_fn = lambda x, y: np.linalg.norm(x - y)
    doo_tree = BinaryDOOTree(domain, explr_p, distance_fn)

    evaled_x = []
    evaled_y = []
    max_y = []
    times = []
    stime = time.time()
    for i in range(n_iter):
        next_node = doo_tree.get_next_node_to_evaluate()
        x_to_evaluate = next_node.x_value
        fval = obj_fcn(x_to_evaluate)
        doo_tree.expand_node(fval, next_node)
        evaled_x.append(x_to_evaluate)
        evaled_y.append(fval)
        max_y.append(np.max(evaled_y))
        times.append(time.time()-stime)
    return evaled_x, evaled_y, max_y, times


def select_epsilon(algorithm, obj_fcn):
    if algorithm.__name__ == 'voo':
        epsilons = [0.1, 0.2, 0.3, 0.4, 0.5]
    elif algorithm.__name__ == 'doo':
        epsilons = [0.01, 0.1, 1, 2, 3]
    elif algorithm.__name__ == 'gpucb':
        epsilons = [0.1, 0.2, 0.3]
    else:
        epsilons = [0]

    max_ys = []
    time_takens = []

    for epsilon in epsilons:
        evaled_x, evaled_y, max_y, time_taken = algorithm(epsilon, obj_fcn)
        max_ys.append(max_y)
        time_takens.append(time_taken)

    return epsilons, max_ys, time_takens


def make_obj_fcn():
    A = np.random.rand(NUMMAX, dim_x) * 10
    C = np.random.rand(NUMMAX)
    return lambda x: shekel(x, A, C)[0]


def main():
    color_dict = pickle.load(open('./plotters/color_dict.p', 'r'))
    color_names = color_dict.keys()[1:]

    doo_max_ys = []
    voo_max_ys = []
    unif_max_ys = []
    for i in range(20):
        obj_fcn = make_obj_fcn()
        for algo_idx, algorithm in enumerate([doo, voo, random_search]):
            if algorithm == doo:
                epsilon = 0.01
            elif algorithm == voo:
                epsilon = 0.3
            else:
                epsilon = 0
            evaled_x, evaled_y, max_y, time_taken = algorithm(epsilon, obj_fcn)
            max_y = np.array(max_y)
            if algorithm == doo:
                doo_max_ys.append(max_y)
            elif algorithm == voo:
                voo_max_ys.append(max_y)
            else:
                unif_max_ys.append(max_y)

    sns.tsplot(doo_max_ys, range(n_iter), ci=95, condition='doo', color=color_dict[color_names[0]])
    sns.tsplot(voo_max_ys, range(n_iter), ci=95, condition='voo', color=color_dict[color_names[1]])
    sns.tsplot(unif_max_ys, range(n_iter), ci=95, condition='unif', color=color_dict[color_names[2]])
    plt.show()


if __name__ == '__main__':
    main()


