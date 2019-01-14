from deap.benchmarks import shekel
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

problem_idx = int(sys.argv[1])
algo_name = sys.argv[2]
dim_x = int(sys.argv[3])
NUMMAX = 10

np.random.seed(problem_idx)
A = np.random.rand(NUMMAX, dim_x)*10
C = np.random.rand(NUMMAX)
if algo_name !='gpucb' and dim_x == 20:
    n_iter = 1000
else:
    n_iter = 200

def shekel_arg0(sol):
    return shekel(sol, A, C)[0]

domain =np.array( [[0]*dim_x,[10]*dim_x] )


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
        x = gp_optimizer.choose_next_point(evaled_x, evaled_y)
        y = shekel_arg0(x)
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
    for i in range(n_iter):
        x = voo.choose_next_point(evaled_x, evaled_y)
        y = shekel_arg0(x)
        evaled_x.append(x)
        evaled_y.append(y)
        max_y.append(np.max(evaled_y))
        times.append(time.time()-stime)
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
    for i in range(n_iter):
        x= np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()
        y = shekel_arg0(x)
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
    for i in range(n_iter):
        next_node = doo_tree.get_next_node_to_evaluate()
        x_to_evaluate = next_node.x_value
        fval = shekel_arg0(x_to_evaluate)
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
        epsilons = [0, 0.1, 1, 2, 3]
    elif algorithm.__name__ == 'gpucb':
        epsilons = [0.1, 0.2, 0.3]
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

    save_dir = './test_results/function_optimization/'+'dim_'+str(dim_x)+'/'+algo_name+'/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

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


