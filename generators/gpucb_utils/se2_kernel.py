from GPy.kern.src.stationary import Stationary
from GPy.util.linalg import tdot
from GPy import util

from GPy.kern.src.psi_comp import PSICOMP_RBF, PSICOMP_RBF_GPU
from GPy.core import Param
from paramz.transformations import Logexp
from GPy.kern.src.grid_kerns import GridRBF

import numpy as np
import sys

sys.path.append('./mover_library/')
from utils import convert_base_pose_to_se2


class StationaryForSE2Distance(Stationary):

    @staticmethod
    def convert_base_pose_to_polar_coordinate(X):
        return np.apply_along_axis(convert_base_pose_to_se2, 1, X)

    def _unscaled_dist(self, X, X2=None):
        """
        Compute the Euclidean distance between each row of X and X2, or between
        each pair of rows of X if X2 is None.
        """

        """
        Converts the values to polar coordinates
        """
        assert X.shape[-1] == 3

        X = self.convert_base_pose_to_polar_coordinate(X)
        if X2 is not None:
            X2 = self.convert_base_pose_to_polar_coordinate(X2)

        if X2 is None:
            Xsq = np.sum(np.square(X), 1)
            r2 = -2. * tdot(X) + (Xsq[:, None] + Xsq[None, :])
            util.diag.view(r2)[:, ] = 0.  # force diagnoal to be zero: sometime numerically a little negative
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)
        else:
            # X2, = self._slice_X(X2)
            X1sq = np.sum(np.square(X), 1)
            X2sq = np.sum(np.square(X2), 1)
            r2 = -2. * np.dot(X, X2.T) + (X1sq[:, None] + X2sq[None, :])
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)


# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


class RBF_SE2(StationaryForSE2Distance):
    """
    Radial Basis Function kernel, aka squared-exponential, exponentiated quadratic or Gaussian kernel:

    .. math::

       k(r) = \sigma^2 \exp \\bigg(- \\frac{1}{2} r^2 \\bigg)

    """
    _support_GPU = True

    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='rbf', useGPU=False,
                 inv_l=False):
        super(RBF_SE2, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name, useGPU=useGPU)
        if self.useGPU:
            self.psicomp = PSICOMP_RBF_GPU()
        else:
            self.psicomp = PSICOMP_RBF()
        self.use_invLengthscale = inv_l
        if inv_l:
            self.unlink_parameter(self.lengthscale)
            self.inv_l = Param('inv_lengthscale', 1. / self.lengthscale ** 2, Logexp())
            self.link_parameter(self.inv_l)

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(RBF_SE2, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.RBF_SE2"
        input_dict["inv_l"] = self.use_invLengthscale
        if input_dict["inv_l"] == True:
            input_dict["lengthscale"] = np.sqrt(1 / float(self.inv_l))
        return input_dict

    def K_of_r(self, r):
        return self.variance * np.exp(-0.5 * r ** 2)

    def dK_dr(self, r):
        return -r * self.K_of_r(r)

    def dK2_drdr(self, r):
        return (r ** 2 - 1) * self.K_of_r(r)

    def dK2_drdr_diag(self):
        return -self.variance  # as the diagonal of r is always filled with zeros

    def __getstate__(self):
        dc = super(RBF_SE2, self).__getstate__()
        if self.useGPU:
            dc['psicomp'] = PSICOMP_RBF()
            dc['useGPU'] = False
        return dc

    def __setstate__(self, state):
        self.use_invLengthscale = False
        return super(RBF_SE2, self).__setstate__(state)

    def spectrum(self, omega):
        assert self.input_dim == 1  # TODO: higher dim spectra?
        return self.variance * np.sqrt(2 * np.pi) * self.lengthscale * np.exp(-self.lengthscale * 2 * omega ** 2 / 2)

    def parameters_changed(self):
        if self.use_invLengthscale: self.lengthscale[:] = 1. / np.sqrt(self.inv_l + 1e-200)
        super(RBF_SE2, self).parameters_changed()

    def get_one_dimensional_kernel(self, dim):
        """
        Specially intended for Grid regression.
        """
        oneDkernel = GridRBF(input_dim=1, variance=self.variance.copy(), originalDimensions=dim)
        return oneDkernel

    # ---------------------------------------#
    #             PSI statistics            #
    # ---------------------------------------#

    def psi0(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[0]

    def psi1(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[1]

    def psi2(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior, return_psi2_n=False)[2]

    def psi2n(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior, return_psi2_n=True)[2]

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        dL_dvar, dL_dlengscale = self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z,
                                                                        variational_posterior)[:2]
        self.variance.gradient = dL_dvar
        self.lengthscale.gradient = dL_dlengscale
        if self.use_invLengthscale:
            self.inv_l.gradient = dL_dlengscale * (self.lengthscale ** 3 / -2.)

    def gradients_Z_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[2]

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[3:]

    def update_gradients_diag(self, dL_dKdiag, X):
        super(RBF_SE2, self).update_gradients_diag(dL_dKdiag, X)
        if self.use_invLengthscale: self.inv_l.gradient = self.lengthscale.gradient * (self.lengthscale ** 3 / -2.)

    def update_gradients_full(self, dL_dK, X, X2=None):
        super(RBF_SE2, self).update_gradients_full(dL_dK, X, X2)
        if self.use_invLengthscale: self.inv_l.gradient = self.lengthscale.gradient * (self.lengthscale ** 3 / -2.)
