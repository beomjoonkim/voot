from sampling_strategies.analytical_voo import AnalyticalVOO
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import ConvexHull

import numpy as np
import time
import matplotlib.pyplot as plt

dist_fcn = lambda x, y: np.linalg.norm(x - y)
voo = AnalyticalVOO(lb=[-2.51] * 10, ub=[2.51] * 10, dist_fcn=dist_fcn)

# evaled_values = np.array([[-3,0],[3,0],[0,3][0,-3]])
evaled_points = [[-3, 0], [3, 0], [0, 3], [0, -3]]

points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
vor = Voronoi(points)
voronoi_plot_2d(vor)
print vor.regions
print vor.points
print vor.vertices
plt.savefig('**************.png')

evaled_points =[]
for i in range(500):
    # new_point = voo.sample_next_point(evaled_values)
    new_point = np.random.uniform([-2.51] * 10, [2.51] * 10)
    evaled_points.append(new_point)

    if len(evaled_points) >= 12:
        stime = time.time()
        vor = Voronoi(evaled_points)
        print "qhull creation time ", time.time() - stime
        print len(evaled_points)
