import numpy as np
import GPy


class StandardContinuousGP:
    def __init__(self, xdim):
        self.model = None
        self.xdim = xdim
        self.kern = GPy.kern.RBF(xdim, variance=5)  # this is tuned anyways with empirical Bayes

    def predict(self, x):
        if self.model is None:
            return np.array([1]), np.array([1])
        if len(x.shape) == 1: x = x[None, :]
        mu, sig = self.model.predict(x)
        return mu, sig

    def update(self, evaled_x, evaled_y, update_hyper_params=True, is_bamsoo=False):
        # todo make the same distance used in the voo
        if len(evaled_x) == 0:
            return
        self.evaled_x = evaled_x
        self.evaled_y = evaled_y
        evaled_x = np.array(evaled_x)
        evaled_y = np.array(evaled_y)[:, None]
        if is_bamsoo:
            # this is for BamSOO, which returns NaN when normalize with a single data point
            if len(evaled_x) == 1:
                normalizer = False  # what is the impact of this?
            else:
                normalizer = True
        else:
            normalizer = True

        print normalizer, update_hyper_params
        self.model = GPy.models.GPRegression(evaled_x, evaled_y, kernel=self.kern, normalizer=normalizer)

        if update_hyper_params:
            self.model.optimize(messages=False, max_f_eval=1000)


class AddStandardContinuousGP:
    def __init__(self, xdim, n_decompositions=None):
        self.n_decompositions = n_decompositions
        self.xdim = xdim
        if xdim == 10:
            self.decompositions = [3, 3, 3, 1]
        elif xdim == 20:
            self.decompositions = [4, 4, 4, 4, 4]
        else:
            raise NotImplementedError
        self.gps = [StandardContinuousGP(d) for d in self.decompositions]
        self.evaled_xs = [[] for _ in range(len(self.decompositions))]
        self.evaled_ys = [[] for _ in range(len(self.decompositions))]

    def update(self, evaled_x, evaled_y, update_hyper_params):
        if len(evaled_x) == 0:
            return

        new_x = evaled_x[-1]
        new_y = evaled_y[-1]
        if self.xdim == 10:
            decomposed_x = [new_x[0:3], new_x[3:6], new_x[6:9], [new_x[-1]]]
        elif self.xdim == 20:
            decomposed_x = [new_x[0:4], new_x[4:8], new_x[8:12], new_x[12:16], new_x[16:20]]
        else:
            raise NotImplementedError

        for idx, x_ in enumerate(decomposed_x):
            self.evaled_xs[idx].append(x_)
            self.evaled_ys[idx].append(new_y)
            self.gps[idx].update(self.evaled_xs[idx], self.evaled_ys[idx], update_hyper_params)


