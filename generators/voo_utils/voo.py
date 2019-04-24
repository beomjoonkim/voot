import numpy as np


class VOO:
    def __init__(self, domain, explr_p, distance_fn=None):
        self.domain = domain
        self.dim_x = domain.shape[-1]
        self.explr_p = explr_p
        if distance_fn is None:
            self.distance_fn = lambda x, y: np.linalg.norm(x-y)

        self.GAUSSIAN = False
        self.UNIFORM_TOUCHING_BOUNDARY = False
        self.DECREASING_UNIFORM = False

    def sample_next_point(self, evaled_x, evaled_y):
        if len(evaled_x) < 1:
            is_sample_from_best_v_region = False
        else:
            rnd = np.random.random()  # this should lie outside
            is_sample_from_best_v_region = rnd < (1 - self.explr_p)
        if is_sample_from_best_v_region:
            x = self.sample_from_best_voronoi_region(evaled_x, evaled_y)
        else:
            x = self.sample_from_uniform()
        return x

    def choose_next_point(self, evaled_x, evaled_y):
        return self.sample_next_point(evaled_x, evaled_y)

    def sample_from_best_voronoi_region(self, evaled_x, evaled_y):
        print "Sampling from best V"
        best_dist = np.inf
        other_dists = np.array([-1])
        counter = 1

        best_evaled_x_idxs = np.argwhere(evaled_y == np.amax(evaled_y))
        best_evaled_x_idxs = best_evaled_x_idxs.reshape((len(best_evaled_x_idxs,)))
        best_evaled_x_idx = best_evaled_x_idxs[0] #np.random.choice(best_evaled_x_idxs)
        best_evaled_x = evaled_x[best_evaled_x_idx]
        other_best_evaled_xs = evaled_x


        closest_best_dist = np.inf
        while np.any(best_dist > other_dists):
            if self.GAUSSIAN:
                variance = (self.domain[1] - self.domain[0]) / np.exp(counter)
                new_x = np.random.normal(best_evaled_x, variance)
                while np.any(new_x > self.domain[1]) or np.any(new_x < self.domain[0]):
                    #print "Edge detecting, sampling other points"
                    new_x = np.random.normal(best_evaled_x, variance)
            elif self.UNIFORM_TOUCHING_BOUNDARY:
                dim_x = self.domain[1].shape[-1]
                possible_range = (self.domain[1] - self.domain[0]) / np.exp(counter)
                possible_values = np.random.uniform(-possible_range, possible_range, (dim_x,))
                new_x = best_evaled_x + possible_values
                while np.any(new_x > self.domain[1]) or np.any(new_x < self.domain[0]):
                    possible_values = np.random.uniform(-possible_range, possible_range, (dim_x,))
                    new_x = best_evaled_x + possible_values
            elif self.DECREASING_UNIFORM:
                dim_x = self.domain[1].shape[-1]
                possible_max = (self.domain[1] - best_evaled_x) / np.exp(counter)
                possible_min = (self.domain[0] - best_evaled_x) / np.exp(counter)
                possible_values = np.random.uniform(possible_min, possible_max, (dim_x,))
                new_x = best_evaled_x + possible_values
                while np.any(new_x > self.domain[1]) or np.any(new_x < self.domain[0]):
                    possible_values = np.random.uniform(possible_min, possible_max, (dim_x,))
                    new_x = best_evaled_x + possible_values
            else:
                dim_x = self.domain[1].shape[-1]
                new_x = np.random.uniform(self.domain[0], self.domain[1], (dim_x,))

            best_dist = self.distance_fn(new_x, best_evaled_x)
            other_dists = np.array([self.distance_fn(other, new_x) for other in other_best_evaled_xs])
            counter += 1
            if closest_best_dist > best_dist:
                closest_best_dist = best_dist
                best_new_x = new_x
            if counter > 1000:
                new_x = best_new_x
                print "Counter reached, switching to Gaussian"
                self.GAUSSIAN = True
                break

        return new_x

    def sample_from_uniform(self):
        dim_parameters = self.domain.shape[-1]
        domain_min = self.domain[0]
        domain_max = self.domain[1]
        return np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()




