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

#A = [[0.5, 0.5], [0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
#C = [0.002, 0.005, 0.005, 0.005, 0.005]

NUMMAX = 10
dim_x = 5
A = np.random.rand(NUMMAX, dim_x)*10
C = np.random.rand(NUMMAX)


def shekel_arg0(sol):
    return shekel(sol, A, C)[0]

domain =np.array( [[0]*dim_x,[10]*dim_x] )


def gpucb():
    explr_p = 0.3
    gp = StandardContinuousGP(dim_x)
    acq_fcn = UCB(zeta=explr_p, gp=gp)
    gp_format_domain = Domain(0, domain)
    gp_optimizer = BO(gp, acq_fcn, gp_format_domain)  # this depends on the problem

    evaled_x = []
    evaled_y = []
    max_y = []
    for i in range(100):
        x = gp_optimizer.choose_next_point(evaled_x, evaled_y)
        y = shekel_arg0(x)
        evaled_x.append(x)
        evaled_y.append(y)
        max_y.append(np.max(evaled_y))

    return evaled_x, evaled_y, max_y


def voo():
    explr_p = 0.3
    evaled_x = []
    evaled_y = []
    max_y = []
    voo = VOO(domain, explr_p)

    for i in range(100):
        print i
        x = voo.choose_next_point(evaled_x, evaled_y)
        y = shekel_arg0(x)
        evaled_x.append(x)
        evaled_y.append(y)
        max_y.append(np.max(evaled_y))
    return evaled_x, evaled_y, max_y


def doo():
    distance_fn = lambda x, y: np.linalg.norm(x - y)
    explr_p = 1
    doo_tree = BinaryDOOTree(domain, explr_p, distance_fn)

    evaled_x = []
    evaled_y = []
    max_y = []
    for i in range(100):
        next_node = doo_tree.get_next_node_to_evaluate()
        x_to_evaluate = next_node.x_value
        fval = shekel_arg0(x_to_evaluate)
        doo_tree.expand_node(fval, next_node)
        evaled_x.append(x_to_evaluate)
        evaled_y.append(fval)
        max_y.append(np.max(evaled_y))

    return evaled_x, evaled_y, max_y


def main():
    color_dict = pickle.load(open('./plotters/color_dict.p', 'r'))
    color_names = color_dict.keys()[1:]

    for algo_idx, algorithm in enumerate([doo, voo, gpucb]):
        evaled_x, evaled_y, max_y = algorithm()
        plt.plot(max_y)
        plot = sns.tsplot(max_y, range(100), ci=95, condition=algorithm.__name__, color=color_dict[color_names[algo_idx]])
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()


"""

fig = plt.figure()
# ax = Axes3D(fig, azim = -29, elev = 50)
ax = Axes3D(fig)
X = np.arange(0, 10, 0.1)
Y = np.arange(0, 10, 0.1)
X, Y = np.meshgrid(X, Y)
Z = np.zeros(X.shape)

for i in xrange(X.shape[0]):
    for j in xrange(X.shape[1]):
        Z[i, j] = shekel_arg0((X[i, j], Y[i, j]))

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, norm=LogNorm(), cmap=cm.jet, linewidth=0.2)

plt.xlabel("x")
plt.ylabel("y")

plt.show()
"""

