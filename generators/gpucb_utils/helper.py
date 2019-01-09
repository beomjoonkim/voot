# Author: Zi Wang
import numpy as np
import scipy.optimize
from sklearn.utils import shuffle
import cPickle as pickle
from sklearn.metrics import confusion_matrix
import os
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import truncnorm
from functools import partial

EPS = 1e-4


def global_minimize(f, fg, domain, n, guesses=None, callback=None):
    if domain.space_type == domain.DISCRETE:
        ty = f(domain.domain)
        idx = ty.argmin()
        return domain.domain[idx], ty[idx]
    else:
        x_range = domain.domain
        dx = x_range.shape[1]
        tx = np.random.uniform(x_range[0], x_range[1], (int(n), dx))
        if guesses is not None:
            tx = np.vstack((tx, guesses))
        ty = f(tx)
        x0 = tx[ty.argmin()]  # 2d array 1*dx
        # if fg is None:
        res = minimize(f, x0, bounds=x_range.T, method='L-BFGS-B', callback=None)
        x_star, y_star = res.x, res.fun
        return x_star, y_star

