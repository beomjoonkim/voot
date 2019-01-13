import numpy as np

class VOO:
    def __init__(self, domain, explr_p, distance_fn=None):
        self.domain = domain
        self.explr_p = explr_p
        if distance_fn is None:
            self.distance_fn = lambda x, y: np.linalg.norm(x-y)

    def choose_next_point(self, evaled_x, evaled_y):
        rnd = np.random.random() # this should lie outside
        is_sample_from_best_v_region = rnd < 1 - self.explr_p and len(evaled_x) > 1
        if is_sample_from_best_v_region:
            x = self.sample_from_best_voronoi_region(evaled_x,evaled_y)
        else:
            x = self.sample_from_uniform()
        return x

    def sample_from_best_voronoi_region(self, evaled_x, evaled_y):
        best_dist = np.inf
        other_dists = np.array([-1])
        counter = 1

        best_evaled_x_idxs = np.argwhere(evaled_y == np.amax(evaled_y))
        best_evaled_x_idxs = best_evaled_x_idxs.reshape((len(best_evaled_x_idxs,)))
        best_evaled_x_idx = np.random.choice(best_evaled_x_idxs)
        best_evaled_x = evaled_x[best_evaled_x_idx]
        other_best_evaled_xs = evaled_x

        while np.any(best_dist > other_dists):
            variance = 0.5*(self.domain[1] - self.domain[0]) / (counter+len(other_dists))
            new_x = np.random.normal(best_evaled_x, variance)
            new_x = np.clip(new_x, self.domain[0], self.domain[1])

            best_dist = self.distance_fn(new_x, best_evaled_x)
            other_dists = np.array([self.distance_fn(other, new_x) for other in other_best_evaled_xs])
            counter+=1
        return new_x

    def sample_from_uniform(self):
        dim_parameters = self.domain.shape[-1]
        domain_min = self.domain[0]
        domain_max = self.domain[1]
        return np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()




