from deap.benchmarks import shekel
from deap import benchmarks
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import cm

from generators.gpucb_utils.gp import StandardContinuousGP
from generators.gpucb_utils.functions import UCB, Domain
from generators.gpucb_utils.bo import BO
from generators.voo_utils.voo import VOO
from generators.doo_utils.doo_tree import BinaryDOOTree

import seaborn as sns
import pickle
import time
import sys
import os
import socket

problem_idx = int(sys.argv[1])
algo_name = sys.argv[2]
dim_x = int(sys.argv[3])
n_fcn_evals = int(sys.argv[4])
obj_fcn = sys.argv[5]
NUMMAX = 10

if obj_fcn == 'shekel':
    np.random.seed(problem_idx)
    A = np.random.rand(NUMMAX, dim_x)*10
    C = np.random.rand(NUMMAX)

if obj_fcn == 'shekel':
    domain =np.array([[0.]*dim_x, [10.]*dim_x])
elif obj_fcn == 'schwefel':
    domain = np.array([[-500.]*dim_x, [500.]*dim_x])
elif obj_fcn == 'rastrigin':
    domain = np.array([[-5.12]*dim_x, [5.12]*dim_x])
elif obj_fcn == 'ackley':
    domain = np.array([[-15.]*dim_x, [30.]*dim_x])
else:
    domain = np.array([[-600.]*dim_x, [600.]*dim_x])


def get_objective_function(sol):
    if obj_fcn == 'shekel':
        return shekel(sol, A, C)[0]
    elif obj_fcn == 'schwefel':
        return -benchmarks.schwefel(sol)[0]
    elif obj_fcn == 'griewank':
        return benchmarks.griewank(sol)[0]
    elif obj_fcn == 'rastrigin':
        return -benchmarks.rastrigin(sol)[0]
    elif obj_fcn == 'ackley':
        return -benchmarks.rastrigin(sol)[0]
    else:
        print "wrong function name"
        sys.exit(-1)


def gpucb(explr_p):
    gp = StandardContinuousGP(dim_x)
    acq_fcn = UCB(zeta=explr_p, gp=gp)
    gp_format_domain = Domain(0, domain)
    gp_optimizer = BO(gp, acq_fcn, gp_format_domain)  # note: this minimizes the negative acq_fcn

    evaled_x = []
    evaled_y = []
    max_y = []
    times = []
    stime = time.time()
    for i in range(n_fcn_evals):
        print 'gp iteration ', i
        x = gp_optimizer.choose_next_point(evaled_x, evaled_y)
        y = get_objective_function(x)
        evaled_x.append(x)
        evaled_y.append(y)
        max_y.append(np.max(evaled_y))
        times.append(time.time()-stime)

    return evaled_x, evaled_y, max_y, times


def voo(explr_p):
    evaled_x = []
    evaled_y = []
    max_y = []
    voo = VOO(domain, explr_p)
    times = []
    stime = time.time()

    for i in range(n_fcn_evals):
        print "%d / %d" % (i, n_fcn_evals)
        if i > 0:
            print 'max value is ', np.max(evaled_y)
        x = voo.choose_next_point(evaled_x, evaled_y)
        if len(x.shape) == 0:
            x = np.array([x])
        y = get_objective_function(x)
        evaled_x.append(x)
        evaled_y.append(y)
        max_y.append(np.max(evaled_y))
        times.append(time.time()-stime)
    print "Max value found", np.max(evaled_y)
    return evaled_x, evaled_y, max_y, times


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


def doo(explr_p):
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
    return evaled_x, evaled_y, max_y, times


def try_many_epsilons(algorithm):
    if algorithm.__name__ == 'voo':
        epsilons = [0.1, 0.2, 0.3, 0.4, 0.5]
    elif algorithm.__name__ == 'doo':
        epsilons = [1, 0.1, 5, 10, 30]
    elif algorithm.__name__ == 'gpucb':
        epsilons = [1, 0.1, 5, 10, 30]
    else:
        epsilons = [0]

    max_ys = []
    time_takens = []
    for epsilon in epsilons:
        evaled_x, evaled_y, max_y, time_taken = algorithm(epsilon)
        max_ys.append(max_y)
        time_takens.append(time_taken)
    return epsilons, max_ys, time_takens


def main():
    hostname = socket.gethostname()
    if hostname == 'dell-XPS-15-9560' or hostname == 'phaedra':
        save_dir = './test_results/function_optimization/' + obj_fcn + '/dim_' + str(dim_x) + '/'+algo_name+'/'
    else:
        save_dir = '/data/public/rw/pass.port/gtamp_results/test_results/function_optimization/' + \
                   obj_fcn + '/dim_' + str(dim_x) + '/' + algo_name+'/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if os.path.isfile(save_dir+'/'+str(problem_idx)+'.pkl'):
        print "Already done"
        return

    if algo_name == 'uniform':
        algorithm = random_search
    elif algo_name == 'voo':
        algorithm = voo
    elif algo_name == 'doo':
        algorithm = doo
    elif algo_name == 'gpucb':
        algorithm = gpucb
    else:
        print "Wrong algo name"
        return

    epsilons, max_ys, time_takens = try_many_epsilons(algorithm)
    pickle.dump({"epsilons": epsilons, 'max_ys': max_ys, 'time_takens': time_takens},
                open(save_dir+'/'+str(problem_idx)+'.pkl', 'wb'))

if __name__ == '__main__':
    main()


