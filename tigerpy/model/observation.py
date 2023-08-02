"""
Model matrices.
"""

import numpy as np
import jax.numpy as jnp
from scipy.interpolate import BSpline as bs

from .model import (
    Array,
    Any)

class Obs:
    """
    Observations.
    """
    def __init__(self, name: str, intercept: bool = True):
        self.name = name
        self.intercept = intercept
        self.design_matrix = None
        self.fixed_incl = False
        self.fixed_data = None
        self.fixed_dim = None
        self.smooth_incl = False
        self.smooth_data = []
        self.smooth_dim = []
        self.smooth_knots = []
        self.smooth_rwk_order = []
        self.smooth_pen_mat = []

    def fixed(self, data: Array, data_copy: bool = True) -> None:
        """
        Method to define the fixed covariates.

        Args:
            data (Array): Array that contains the fixed covariates (excluding the intercept).
            data_copy (bool, optional): Should the array be copied. Defaults to True.
        """

        self.data_copy = data_copy

        if type(data) is not np.array:
            data = np.asarray(data, dtype=np.float32)

        if self.data_copy:
            self.fixed_data = data.copy()
        else:
            self.fixed_data = data

        if self.intercept:
            self.fixed_data = np.column_stack((np.ones(len(self.fixed_data)), self.fixed_data))

        if self.design_matrix is None:
            self.design_matrix = jnp.asarray(self.fixed_data, dtype=jnp.float32)
            self.fixed_dim = self.fixed_data.shape[1]
            self.fixed_incl = True
        else:
            print("Design-Matrix contains already smooth terms")
            raise ValueError("Fixed covariate effects must be defined first.")

    def smooth(self, data: Array, n_knots=20, degree=3, rwk=2) -> None:
        """
        Method to define smooth B-Spline covariates.

        Args:
            data (Array): 1D-Array that contains a covariate.
            n_knots (int, optional): Number of Knots . Defaults to 20.
            degree (int, optional): The degree of the B-spline. Defaults to 3 (cubic).
            rwk (int, optional): Random walk order that defines the penalisation of the coefficients. Defaults to 2 (second order).
        """

        # use n_knots+k+1 knots to have dim (n, n_knots) in the spline design_matrix
        knots = np.linspace(data.min(), data.max(), num=n_knots+degree+1)

        smooth_matrix = bs.design_matrix(data, t=knots, k=degree, extrapolate=True).toarray()
        self.smooth_data.append(smooth_matrix)
        self.smooth_dim.append(smooth_matrix.shape[1])
        self.smooth_knots.append(knots)
        self.smooth_rwk_order.append(rwk)

        # create the penalty matrix
        diff_mat = jnp.diff(jnp.eye(smooth_matrix.shape[1]), n=rwk, axis=0)
        pen = jnp.dot(diff_mat.T, diff_mat)
        self.smooth_pen_mat.append(pen)

        if self.design_matrix is None:
            if self.intercept is False:
                self.design_matrix = jnp.asarray(smooth_matrix, dtype=jnp.float32)
                self.smooth_incl = True
            else:
                self.fixed_data = np.ones((smooth_matrix.shape[0],1))
                self.fixed_dim = 1
                self.fixed_incl = True
                self.smooth_incl = True
        else:
            if self.smooth_incl is None:
                self.smooth_incl = True

    def center(self) -> Array:
        if self.smooth_incl is True and self.fixed_incl is True:
            fixed = jnp.asarray(self.fixed_data, dtype=jnp.float32)
            self.design_mat_cent= [fixed]
            self.smooth_pen_mat_cent = []
            self.smooth_dim_cent = []
            self.orth_factor = []

            for i in range(len(self.smooth_data)):
                smooth_mat = self.smooth_data[i]
                smooth_mat = jnp.asarray(smooth_mat, dtype=jnp.float32)
                c = jnp.mean(smooth_mat, axis=0)
                c = jnp.expand_dims(c, axis=1)
                Q, _ = jnp.linalg.qr(c, mode="complete")

                Tb = Q[:,1:]
                smooth_mat_new = jnp.dot(smooth_mat, Tb)
                pen_new = Tb.T @ self.smooth_pen_mat[i] @ Tb

                self.design_mat_cent.append(smooth_mat_new)
                self.smooth_pen_mat_cent.append(pen_new)
                self.smooth_dim_cent.append(self.smooth_dim[i]-1)
                self.orth_factor.append(Tb)

            self.design_matrix = jnp.concatenate(self.design_mat_cent, axis=1)
        else:
            print("No need to include identifiability constraints.")
            raise ValueError("The model must at least contain one fixed parameter.")

# https://stats.stackexchange.com/questions/517375/splines-relationship-of-knots-degree-and-degrees-of-freedom
