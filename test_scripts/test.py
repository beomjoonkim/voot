import numpy as np
import GPy
import time

def toy_ARD(max_iters=1000, kernel_type='linear', num_samples=300, D=4, optimize=True, plot=True):
    # Create an artificial dataset where the values in the targets (Y)
    # only depend in dimensions 1 and 3 of the inputs (X). Run ARD to
    # see if this dependency can be recovered
    X1 = np.sin(np.sort(np.random.rand(num_samples, 1) * 10, 0))
    X2 = np.cos(np.sort(np.random.rand(num_samples, 1) * 10, 0))
    X3 = np.exp(np.sort(np.random.rand(num_samples, 1), 0))
    X4 = np.log(np.sort(np.random.rand(num_samples, 1), 0))
    X = np.hstack((X1, X2, X3, X4))

    Y1 = np.asarray(2 * X[:, 0] + 3).reshape(-1, 1)
    Y2 = np.asarray(4 * (X[:, 2] - 1.5 * X[:, 0])).reshape(-1, 1)
    Y = np.hstack((Y1, Y2))

    Y = np.dot(Y, np.random.rand(2, D));
    Y = Y + 0.2 * np.random.randn(Y.shape[0], Y.shape[1])
    Y -= Y.mean()
    Y /= Y.std()

    if kernel_type == 'linear':
        kernel = GPy.kern.Linear(X.shape[1], ARD=1)
    elif kernel_type == 'rbf_inv':
        kernel = GPy.kern.RBF_inv(X.shape[1], ARD=1)
    else:
        kernel = GPy.kern.RBF(X.shape[1], ARD=1)
    kernel += GPy.kern.White(X.shape[1]) + GPy.kern.Bias(X.shape[1])


    stime=time.time()
    m = GPy.models.GPRegression(X, Y, kernel)

    # len_prior = GPy.priors.inverse_gamma(1,18) # 1, 25
    # m.set_prior('.*lengthscale',len_prior)

    m.optimize(optimizer='scg', max_iters=max_iters)
    print time.time()-stime
    return time.time() - stime


ttt=0
for i in range(1,1000):
    ttt += toy_ARD(num_samples=i)
    print i, ttt
print ttt