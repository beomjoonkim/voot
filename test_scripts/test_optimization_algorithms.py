import pickle
import time
import sys
import os
import argparse
import numpy as np
import random

from deap.benchmarks import shekel
from deap import benchmarks

import socket

if socket.gethostname() == 'dell-XPS-15-9560' or socket.gethostname() == 'lab':
    import pygmo as pg

if True:
    from generators.gpucb_utils.gp import StandardContinuousGP, AddStandardContinuousGP
    from generators.gpucb_utils.functions import UCB, Domain, AddUCB
    from generators.gpucb_utils.bo import BO, AddBO

from generators.soo_utils.bamsoo_tree import BamBinarySOOTree
from generators.voo_utils.voo import VOO
from generators.doo_utils.doo_tree import BinaryDOOTree
from generators.soo_utils.soo_tree import BinarySOOTree

parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('-problem_idx', type=int, default=0)
parser.add_argument('-algo_name', type=str, default='stosoo')
parser.add_argument('-obj_fcn', type=str, default='griewank')
parser.add_argument('-dim_x', type=int, default=10)
parser.add_argument('-n_fcn_evals', type=int, default=500)
parser.add_argument('-voo_sampling_mode', type=str, default='centered_uniform')
parser.add_argument('-switch_counter', type=int, default=100)
parser.add_argument('-low_dim', type=int, default=2)
args = parser.parse_args()

problem_idx = args.problem_idx
algo_name = args.algo_name
dim_x = args.dim_x
n_fcn_evals = args.n_fcn_evals
obj_fcn = args.obj_fcn

np.random.seed(problem_idx)
random.seed(problem_idx)

NUMMAX = 10
if obj_fcn == 'shekel':

    # np.random.seed(problem_idx)
    # A = np.random.rand(NUMMAX, dim_x)*10
    # C = np.random.rand(NUMMAX)

    if args.dim_x == 2:
        A = np.array([[
            0.5, 0.5],
            [0.25, 0.25],
            [0.25, 0.75],
            [0.75, 0.25],
            [0.75, 0.75]
        ]) * 500
        C = np.array([0.002, 0.005, 0.005, 0.005, 0.005]) * 500
    else:
        config = pickle.load(
            open('./test_results/function_optimization/shekel/shekel_dim_' + str(args.dim_x) + '.pkl', 'r'))
        A = config['A']
        C = config['C']

if obj_fcn == 'shekel':
    domain = np.array([[-500.] * dim_x, [500.] * dim_x])
elif obj_fcn == 'schwefel':
    domain = np.array([[-500.] * dim_x, [500.] * dim_x])
elif obj_fcn == 'rastrigin':
    domain = np.array([[-5.12] * dim_x, [5.12] * dim_x])
elif obj_fcn == 'ackley':
    domain = np.array([[-30.] * dim_x, [30.] * dim_x])
elif obj_fcn == 'griewank':
    domain = np.array([[-600.] * dim_x, [600.] * dim_x])
elif obj_fcn == 'rosenbrock':
    domain = np.array([[-100.] * dim_x, [100.] * dim_x])
elif obj_fcn == 'schaffer':
    domain = np.array([[-100.] * dim_x, [100.] * dim_x])
else:
    raise NotImplementedError


def get_objective_function(sol):
    if obj_fcn == 'shekel':
        return shekel(sol, A, C)[0]
    elif obj_fcn == 'schwefel':
        return -benchmarks.schwefel(sol)[0]
    elif obj_fcn == 'griewank':
        return -benchmarks.griewank(sol)[0]
    elif obj_fcn == 'rastrigin':
        return -benchmarks.rastrigin(sol)[0]
    elif obj_fcn == 'ackley':
        return -benchmarks.ackley(sol)[0]
    elif obj_fcn == 'rosenbrock':
        return -benchmarks.rosenbrock(sol)[0]
    elif obj_fcn == 'schaffer':
        return -benchmarks.schaffer(sol)[0]
    else:
        print "wrong function name"
        sys.exit(-1)


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
        # if i == 0:
        #    x = (domain_min+domain_max)/2.0
        # else:
        x = np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()
        if len(x.shape) == 0:
            x = np.array([x])
        y = get_objective_function(x)
        evaled_x.append(x)
        evaled_y.append(y)
        max_y.append(np.max(evaled_y))
        times.append(time.time() - stime)
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
        print "%d / %d" % (i, n_fcn_evals)
        if i > 0:
            print 'max value is ', np.max(evaled_y)
        next_node = doo_tree.get_next_point_and_node_to_evaluate()
        x_to_evaluate = next_node.cell_mid_point
        next_node.evaluated_x = x_to_evaluate
        fval = get_objective_function(x_to_evaluate)
        doo_tree.expand_node(fval, next_node)

        evaled_x.append(x_to_evaluate)
        evaled_y.append(fval)
        max_y.append(np.max(evaled_y))
        times.append(time.time() - stime)

    best_idx = np.where(evaled_y == max_y[-1])[0][0]
    print evaled_x[best_idx], evaled_y[best_idx]
    print "Max value found", np.max(evaled_y)
    print "Magnitude", np.linalg.norm(evaled_x[best_idx])
    print "Explr p", explr_p
    return evaled_x, evaled_y, max_y, times


def soo(dummy):
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
        times.append(time.time() - stime)

    print "Max value found", np.max(evaled_y)
    return evaled_x, evaled_y, max_y, times


def bamsoo(explr_p, save_dir):
    bamsoo_tree = BamBinarySOOTree(domain, explr_p)

    evaled_x = []
    evaled_y = []
    max_y = []
    times = []

    stime = time.time()
    for i in range(n_fcn_evals):
        next_node = bamsoo_tree.get_next_point_and_node_to_evaluate()
        x_to_evaluate = next_node.cell_mid_point
        next_node.evaluated_x = x_to_evaluate
        fval = get_objective_function(x_to_evaluate)
        bamsoo_tree.expand_node(fval, next_node, evaled_x, evaled_y)

        evaled_x.append(x_to_evaluate)
        evaled_y.append(fval)
        max_y.append(np.max(evaled_y))
        times.append(time.time() - stime)
        print i, max_y[-1], evaled_x[-1]

    print "Max value found", np.max(evaled_y)
    return evaled_x, evaled_y, max_y, times


def add_gpucb(explr_p, save_dir):
    gp = AddStandardContinuousGP(dim_x)
    acq_fcn = AddUCB(add_gp=gp)
    gp_format_domain = Domain(0, domain)
    gp_optimizer = AddBO(gp, acq_fcn, gp_format_domain)  # note: this minimizes the negative acq_fcn

    evaled_x = []
    evaled_y = []
    max_y = []
    times = []
    stime = time.time()
    for i in range(n_fcn_evals):
        x = gp_optimizer.choose_next_point(evaled_x, evaled_y)
        y = get_objective_function(x)
        evaled_x.append(x)
        evaled_y.append(y)
        max_y.append(np.max(evaled_y))
        times.append(time.time() - stime)
        print 'gp iteration ', i, np.max(evaled_y), x

        pickle.dump({'epsilon': [explr_p], 'max_ys': [max_y]},
                    open(save_dir + '/' + str(problem_idx) + '.pkl', 'wb'))
    return evaled_x, evaled_y, max_y, times


def gpucb(explr_p, save_dir):
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
        x = gp_optimizer.choose_next_point(evaled_x, evaled_y)
        y = get_objective_function(x)
        evaled_x.append(x)
        evaled_y.append(y)
        max_y.append(np.max(evaled_y))
        times.append(time.time() - stime)
        print evaled_x
        print 'gp iteration ', i, np.max(evaled_y)

        pickle.dump({'epsilon': [explr_p], 'max_ys': [max_y]},
                    open(save_dir + '/' + str(problem_idx) + '.pkl', 'wb'))

    return evaled_x, evaled_y, max_y, times


def rembo_gpucb(explr_p, low_dim, save_dir):
    gp = StandardContinuousGP(low_dim)
    acq_fcn = UCB(zeta=explr_p, gp=gp)

    # Generate A
    domain_min = domain[0][0]
    domain_max = domain[1][1]
    original_dim = len(domain[0])
    A = np.random.rand(original_dim, low_dim) * (domain_max-domain_min) + domain_min

    # has to be such that when I multiply it by A, then it should roughly stay within domain_min
    low_dim_domain_min = [-np.sqrt(domain_max)/2.0] * low_dim
    low_dim_domain_max = [np.sqrt(domain_max)/2.0] * low_dim
    low_dim_domain = [low_dim_domain_min, low_dim_domain_max]

    gp_format_domain = Domain(0, low_dim_domain)
    gp_optimizer = BO(gp, acq_fcn, gp_format_domain)  # note: this minimizes the negative acq_fcn

    evaled_x = []
    evaled_y = []

    evaled_low_dim_x = []
    max_y = []
    times = []
    stime = time.time()
    for i in range(n_fcn_evals):
        print 'gp iteration ', i
        low_dim_x = gp_optimizer.choose_next_point(evaled_low_dim_x, evaled_y)
        x = np.dot(A, low_dim_x)

        # keep it in range
        if not(np.all(x <= domain_max)):
            x[np.where(x >= domain_max)] = domain_max
        if not(np.all(x >= domain_min)):
            x[np.where(x <= domain_min)] = domain_min
        y = get_objective_function(x)
        evaled_low_dim_x.append(low_dim_x)
        evaled_x.append(x)
        evaled_y.append(y)
        max_y.append(np.max(evaled_y))
        times.append(time.time() - stime)

        pickle.dump({'epsilon': [explr_p], 'max_ys': [max_y]},
                    open(save_dir + '/' + str(problem_idx) + '.pkl', 'wb'))
        print 'max_y', max_y

    return evaled_x, evaled_y, max_y, times


def voo(explr_p):
    evaled_x = []
    evaled_y = []
    max_y = []
    voo = VOO(domain, explr_p, args.voo_sampling_mode, args.switch_counter)
    times = []
    stime = time.time()
    print 'explr_p', explr_p

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
        times.append(time.time() - stime)

    best_idx = np.where(evaled_y == max_y[-1])[0][0]
    print evaled_x[best_idx], evaled_y[best_idx]
    print "Max value found", np.max(evaled_y)
    print "Magnitude", np.linalg.norm(evaled_x[best_idx])
    print "Explr p", explr_p

    return evaled_x, evaled_y, max_y, times


class GeneticAlgoProblem:
    def fitness(self, x):
        return [-get_objective_function(x)]

    def get_bounds(self):
        return (domain[0].tolist(), domain[1].tolist())


def genetic_algorithm(explr_p):
    prob = pg.problem(GeneticAlgoProblem())
    #sade = pg.sade(gen=1, ftol=1e-20, xtol=1e-20)
    population_size = 10

    if obj_fcn == 'griewank' or dim_x == 3:
        total_evals = 500
    elif obj_fcn == 'shekel' and dim_x == 20:
        total_evals = 5000
    else:
        total_evals = 1000

    generations = total_evals / population_size
    optimizer = pg.cmaes(gen=generations, ftol=1e-20, xtol=1e-20)
    algo = pg.algorithm(optimizer)
    algo.set_verbosity(1)
    pop = pg.population(prob, size=5)
    pop = algo.evolve(pop)
    print pop.champion_f
    champion_x = pop.champion_x
    uda = algo.extract(pg.cmaes)
    log = np.array(uda.get_log())
    n_fcn_evals = log[:, 1]
    pop_best_at_generation = -log[:, 2]
    evaled_x = None
    evaled_y = pop_best_at_generation

    max_y = [pop_best_at_generation[0]]
    for y in pop_best_at_generation[1:]:
        if y > max_y[-1]:
            max_y.append(y)
        else:
            max_y.append(max_y[-1])

    return evaled_x, evaled_y, max_y, 0


def get_exploration_parameters(algorithm):
    if algorithm.__name__.find('voo') != -1:
        epsilons = [0.001]
    elif algorithm.__name__ == 'doo':
        epsilons = [np.finfo(float).eps, 0.0001, 1, 0.1, 0.01, np.finfo(np.float32).eps, 0.0000001, 0.000001, 0.001,
                    0.01]  # this has more initial points
    elif algorithm.__name__ == 'gpucb':
        epsilons = [0.01, 1, 0.1, 5, 10, 30]
    elif algorithm.__name__ == 'rembo_gpucb':
        epsilons = [1, 0.1, 5]
    elif algorithm.__name__ == 'add_gpucb':
        epsilons = [0]
    elif algorithm.__name__ == 'soo':
        epsilons = [0]
    elif algorithm.__name__.find('random_search') != -1 or algorithm.__name__.find('stounif') != -1:
        epsilons = [0]
    elif algorithm.__name__.find('genetic_algorithm') != -1:
        epsilons = [0]
    elif algorithm.__name__ == 'bamsoo':
        epsilons = [0.1, 0.7, 0.9]
    else:
        print algorithm.__name__
        raise NotImplementedError
    return epsilons


def main():
    if socket.gethostname() != 'shakey' and socket.gethostname() != 'phaedra' \
            and socket.gethostname() != 'dell-XPS-15-9560' \
            and socket.gethostname() != 'lab':
        save_dir = '/data/public/rw/pass.port/gtamp_results/test_results/function_optimization/' + obj_fcn + '/dim_' \
                   + str(dim_x) + '/' + algo_name + '/'
    else:
        save_dir = './test_results/function_optimization/' + obj_fcn + '/dim_' + str(dim_x) + '/' + algo_name + '/'

    if not os.path.isdir(save_dir):
        try:
            os.makedirs(save_dir)
        except OSError:
            pass

    if os.path.isfile(save_dir + '/' + str(problem_idx) + '.pkl'):
        print "Already done"
        #return

    if algo_name == 'uniform':
        algorithm = random_search
    elif algo_name == 'voo':
        algorithm = voo
    elif algo_name == 'doo':
        algorithm = doo
    elif algo_name == 'gpucb':
        algorithm = gpucb
    elif algo_name == 'soo':
        algorithm = soo
    elif algo_name == 'cmaes':
        algorithm = genetic_algorithm
    elif algo_name == 'rembo_gpucb':
        algorithm = rembo_gpucb
    elif algo_name == 'bamsoo':
        algorithm = bamsoo
    elif algo_name == 'add_gpucb':
        algorithm = add_gpucb
    else:
        print "Wrong algo name"
        return

    epsilons = get_exploration_parameters(algorithm)

    max_ys = []
    time_takens = []
    for epsilon in epsilons:
        if algo_name == 'gpucb':
            evaled_x, evaled_y, max_y, time_taken = algorithm(epsilon, save_dir)
        elif algo_name == 'rembo_gpucb':
            evaled_x, evaled_y, max_y, time_taken = algorithm(epsilon, args.low_dim, save_dir)
        else:
            evaled_x, evaled_y, max_y, time_taken = algorithm(epsilon, save_dir)

        max_ys.append(max_y)
        time_takens.append(time_taken)

    pickle.dump({"epsilons": epsilons, 'max_ys': max_ys, 'time_takens': time_takens},
                open(save_dir + '/' + str(problem_idx) + '.pkl', 'wb'))

    return epsilons, max_ys, time_takens


if __name__ == '__main__':
    main()
