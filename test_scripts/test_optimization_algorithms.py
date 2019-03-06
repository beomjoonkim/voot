from deap.benchmarks import shekel
from deap import benchmarks
import numpy as np

from generators.gpucb_utils.gp import StandardContinuousGP
from generators.gpucb_utils.functions import UCB, Domain
from generators.gpucb_utils.bo import BO
from generators.voo_utils.voo import VOO
from generators.voo_utils.stovoo import StoVOO
from generators.doo_utils.doo_tree import BinaryDOOTree
from generators.soo_utils.soo_tree import BinarySOOTree

import pickle
import time
import sys
import os
import socket
import argparse

parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('-ucb', type=float, default=1.0)
parser.add_argument('-widening_parameter', type=float, default=0.8)
parser.add_argument('-problem_idx', type=int, default=0)
parser.add_argument('-algo_name', type=str, default='voo')
parser.add_argument('-obj_fcn', type=str, default='ackley')
parser.add_argument('-dim_x', type=int, default=20)
parser.add_argument('-n_fcn_evals', type=int, default=500)
parser.add_argument('-stochastic_objective', action='store_true', default=False)
parser.add_argument('-function_noise', type=float, default=10)
args = parser.parse_args()

problem_idx = args.problem_idx
algo_name = args.algo_name
dim_x = args.dim_x
n_fcn_evals = args.n_fcn_evals
obj_fcn = args.obj_fcn
stochastic_objective = args.stochastic_objective
noise = args.function_noise

ucb_parameter = args.ucb
widening_parameter = args.widening_parameter

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
        return -benchmarks.griewank(sol)[0]
    elif obj_fcn == 'rastrigin':
        return -benchmarks.rastrigin(sol)[0]
    elif obj_fcn == 'ackley':
        return -benchmarks.rastrigin(sol)[0]
    else:
        print "wrong function name"
        sys.exit(-1)


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
        print 'gp iteration ', i
        x = gp_optimizer.choose_next_point(evaled_x, evaled_y)
        y = get_objective_function(x)
        evaled_x.append(x)
        evaled_y.append(y)
        max_y.append(np.max(evaled_y))
        times.append(time.time()-stime)

        pickle.dump({'epsilon':[explr_p], 'max_ys': [max_y]},
                    open(save_dir + '/' + str(problem_idx) + '.pkl', 'wb'))

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


def stovoo(explr_p):
    evaled_x = []
    evaled_y = []
    max_y = []
    stovoo = StoVOO(domain, ucb_parameter, widening_parameter, explr_p)
    times = []

    stime = time.time()
    for i in range(n_fcn_evals):
        print "%d / %d" % (i, n_fcn_evals)
        if i > 0:
            print 'max value is ', np.max(evaled_y)
        evaled_arm = stovoo.choose_next_point(evaled_x, evaled_y)
        y = get_objective_function(evaled_arm.x_value)
        noise_y = y + np.random.normal(0, noise)
        stovoo.update_evaluated_arms(evaled_arm, noise_y)

        # todo what should I record? The expected values?
        evaled_x.append(evaled_arm.x_value)
        evaled_y.append(y)
        max_y.append(np.max(evaled_y))
        times.append(time.time()-stime)

    arm_with_highest_expected_value = stovoo.arms[np.argmax([a.expected_value for a in stovoo.arms])]
    best_arm_x_value = arm_with_highest_expected_value.x_value
    best_arm_true_y = get_objective_function(best_arm_x_value)

    print "Max value found", np.max(evaled_y)
    return evaled_x, evaled_y, max_y, times, best_arm_true_y


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
        # todo run this and see where the bug is. It's not tested fully.

        evaled_x.append(x_to_evaluate)
        evaled_y.append(fval)
        max_y.append(np.max(evaled_y))
        times.append(time.time()-stime)
        print np.max(evaled_y)

    return evaled_x, evaled_y, max_y, times


def get_exploration_parameters(algorithm):
    if algorithm.__name__ == 'voo':
        epsilons = [0.1, 0.2, 0.3, 0.4, 0.5]
    elif algorithm.__name__ == 'doo':
        epsilons = [1, 0.1, 5, 10, 30]
    elif algorithm.__name__ == 'gpucb':
        # schwefel: best epsilon = 0.1 for dimension 10
        #           best_epsilon = 1 for dimension 20
        if obj_fcn == 'schwefel':
            if dim_x == 3:
                epsilons = [0.1]
            elif dim_x == 10:
                epsilons = [1]
            elif dim_x == 20:
                epsilons = [1]
        else:
            epsilons = [1, 0.1, 5, 10, 30]
    else:
        epsilons = [0]

    return epsilons


def main():
    hostname = socket.gethostname()
    if hostname == 'dell-XPS-15-9560' or hostname == 'phaedra':
        if stochastic_objective:
            save_dir = './test_results/stochastic_function_optimization/' + obj_fcn + '/noise_' + str(noise) +\
                       '/ucb_' + str(ucb_parameter) + \
                       '/widening_'+str(widening_parameter) + \
                       '/dim_' + str(dim_x) + '/'+algo_name+'/'
        else:
            save_dir = './test_results/function_optimization/' + obj_fcn + '/dim_' + str(dim_x) + '/'+algo_name+'/'
    else:
        if stochastic_objective:
            save_dir = './test_results/stochastic_function_optimization/' + obj_fcn + '/ucb_' + str(ucb_parameter) + \
                       '/widening_' + str(widening_parameter) + '/dim_' + str(dim_x) + '/' + algo_name + '/'
        else:
            save_dir = '/data/public/rw/pass.port/gtamp_results/test_results/function_optimization/' + \
                       obj_fcn + '/dim_' + str(dim_x) + '/' + algo_name+'/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if os.path.isfile(save_dir+'/'+str(problem_idx)+'.pkl'):
        print "Already done"
        return

    if stochastic_objective:
        if algo_name == 'uniform':
            algorithm = random_search
        elif algo_name == 'stovoo':
            algorithm = stovoo
        else:
            raise NotImplementedError
    else:
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
        else:
            print "Wrong algo name"
            return

    epsilons = get_exploration_parameters(algorithm)

    max_ys = []
    time_takens = []
    for epsilon in epsilons:
        if stochastic_objective:
            evaled_x, evaled_y, max_y, time_taken, best_arm_value = algorithm(epsilon)
        else:
            if algo_name == 'gpucb':
                evaled_x, evaled_y, max_y, time_taken = algorithm(epsilon, save_dir)
            else:
                evaled_x, evaled_y, max_y, time_taken = algorithm(epsilon)

        max_ys.append(max_y)
        time_takens.append(time_taken)

    if stochastic_objective:
        pickle.dump({"epsilons": epsilons, 'max_ys': max_ys, 'time_takens': time_takens,
                     'best_arm_value': best_arm_value},
                    open(save_dir+'/'+str(problem_idx)+'.pkl', 'wb'))
    else:
        pickle.dump({"epsilons": epsilons, 'max_ys': max_ys, 'time_takens': time_takens},
                        open(save_dir+'/'+str(problem_idx)+'.pkl', 'wb'))
    return epsilons, max_ys, time_takens


if __name__ == '__main__':
    main()


