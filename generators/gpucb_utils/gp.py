import numpy as np
import GPy


class StandardContinuousGP():
    def __init__(self, xdim):
        self.model = None
        self.kern = GPy.kern.RBF(xdim, variance=500)

    def predict(self, x):
        if self.model is None:
            return np.array([1]), np.array([1])
        if len(x.shape) == 1: x = x[None, :]
        mu, sig = self.model.predict(x)
        return mu, sig

    def update(self, evaled_x, evaled_y):
        self.evaled_x = np.array(evaled_x)
        self.evaled_y = np.array(evaled_y)
        if len(evaled_x) == 0:
            return
        evaled_x = np.array(evaled_x)
        evaled_y = np.array(evaled_y)[:, None]
        self.model = GPy.models.GPRegression(evaled_x, evaled_y, kernel=self.kern)


