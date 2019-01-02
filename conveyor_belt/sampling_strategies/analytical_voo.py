from scipy.spatial import HalfspaceIntersection
import copy
import numpy as np
import time


class VoronoiRegion:
    def __init__(self, generator, boundary_halfplanes, dist_fcn):
        self.dist_fcn = dist_fcn
        self.generator = generator
        self.halfplanes = copy.deepcopy(boundary_halfplanes)
        self.region = HalfspaceIntersection(np.array(self.halfplanes), self.generator, incremental=True)
        self.update_max_dist()

    def add_halfplane(self, halfplane):
        self.halfplanes.append(halfplane)
        self.region.add_halfspaces(halfplane[None,:])
        self.update_max_dist()

    def update_max_dist(self):
        convexhull = self.region.intersections
        max_dist = -np.inf
        for p2 in convexhull:
            dist = self.dist_fcn(self.generator, p2)
            if dist > max_dist:
                max_dist = dist
        self.max_dist = max_dist

    def get_max_dist(self):
        return self.max_dist

    def is_halfplane_in_region(self, halfplane):
        #for existing_halfplane in self.halfplanes:
        pass

    def filter_halfplanes(self):
        for halfplane in self.halfplanes:
            if not self.is_halfplane_in_region(halfplane):
                self.halfplanes.remove(halfplane)


class AnalyticalVOO:
    def __init__(self, lb, ub, dist_fcn, boundary_halfplanes=None):
        if boundary_halfplanes is None:
            search_space_dim = len(lb)
            self.boundary_halfplanes = self.create_boundary_halfplanes(lb, ub)
        else:
            self.boundary_halfplanes = boundary_halfplanes

        self.voronoi_regions = []
        self.dist_fcn = dist_fcn
        self.lb = lb
        self.ub = ub

    def create_boundary_halfplanes(self, lb, ub):
        search_space_dim = len(lb)
        halfplanes = []
        for i in range(search_space_dim):
            # add lower bound limits
            row = np.zeros((search_space_dim+1,))
            row[i] = -1
            row[-1] = lb[i]
            halfplanes.append(row)

        for i in range(search_space_dim):
            # add lower bound limits
            row = np.zeros((search_space_dim+1,))
            row[i] = 1
            row[-1] = -ub[i]
            halfplanes.append(row)

        return halfplanes

        #self.boundary_halfplanes = [[-1, 0,  -2.51],
        #                            [1, 0, -2.51],
        #                            [0, -1,  -2.51],
        #                            [0, 1,  -2.51]]
    @staticmethod
    def get_perpendicular_bisector(p1, p2):
        diff = p1 - p2
        add = p1 + p2
        b = np.sum(diff * add) / 2.0
        A = np.hstack([diff, -b])

        if np.dot(diff, p1) - b > 0:
            return -A  # scipy wants it like this
        else:
            return A

    @staticmethod
    def base_conf_distance(x, y):
        return np.sum(abs(x - y))


    def get_midpoint_from_generator_and_convex_hull(self, region):
        generator = region.generator
        convexhull = region.region.intersections
        import pdb;pdb.set_trace()

    def random_sample_from_region(self,chosen_point):
        new_point = np.random.uniform(self.lb, self.ub)
        dists_to_non_best_actions = np.array([self.base_conf_distance(new_point, evaled.generator)
                                              for evaled in self.voronoi_regions if np.all(new_point != chosen_point)])
        dist_to_curr_best_action = np.array(self.base_conf_distance(new_point, chosen_point))

        while np.any(dist_to_curr_best_action > dists_to_non_best_actions):
            new_point = np.random.uniform(self.lb, self.ub)
            dists_to_non_best_actions = np.array([self.base_conf_distance(new_point, evaled.generator)
                                                  for evaled in self.voronoi_regions if np.all(new_point != chosen_point)])
            dist_to_curr_best_action = np.array(self.base_conf_distance(new_point, chosen_point))

        return new_point

    def add_voronoi_region(self, new_point):
        new_region = VoronoiRegion(new_point, self.boundary_halfplanes, self.dist_fcn)

        new_region_halfplanes = []
        for evaled in self.voronoi_regions:
            new_halfplane_pointing_new_point = self.get_perpendicular_bisector(new_point, evaled.generator)
            new_halfplane_pointing_existing_point = -new_halfplane_pointing_new_point
            new_region_halfplanes.append(new_halfplane_pointing_new_point)


            new_region.add_halfplane(new_halfplane_pointing_new_point)

            #todo filter the halfplanes that lies outside of the Voroi region

            # check if this halfplane needs to be inserted
            evaled.add_halfplane(new_halfplane_pointing_existing_point)
            #todo filter the halfplanes that lies outside of the Voroi region

        self.voronoi_regions.append(new_region)

    def sample_next_point(self, evaled_values):
        if len(self.voronoi_regions) == 0:
            new_point = np.random.uniform(self.lb, self.ub)
            self.add_voronoi_region(new_point)
            return new_point

        # compute the f(x) + d(x,x_gen) for each region
        max_ub = -np.inf
        stime = time.time()
        for evaled_idx, evaled in enumerate(self.voronoi_regions):
            ub = evaled_values[evaled_idx] + evaled.get_max_dist()
            #print evaled_values[evaled_idx], evaled.get_max_dist()
            if ub >= max_ub:
                max_ub = ub
                best_region = evaled
        print "Upperbound computation time: ",time.time()-stime

        new_point = self.random_sample_from_region(best_region.generator)
        #new_point = get_midpoint_from_generator_and_convex_hull(best_region)

        stime = time.time()
        self.add_voronoi_region(new_point)
        print 'Voronoi region creation time', time.time()-stime
        print 'Number of halfplanes', len(self.voronoi_regions[0].halfplanes)
        print 'Number of points', len(self.voronoi_regions)
        return new_point
