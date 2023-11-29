"""
Design matrices.
"""

import numpy as np
from scipy.interpolate import BSpline as bs

from .model import (
    Array
)

class Obs:
    """
    Class to create design matrices for a linear predictor.
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
            data (Array): An array like object, that can be transformed to an np.ndarray,
            containing the fixed covariates (excluding the intercept).
            data_copy (bool, optional): Should the array be copied. Defaults to True.
        """

        self.data_copy = data_copy

        if type(data) is not np.array:
            data = np.asarray(data, dtype=np.float32)

        if self.data_copy is True:
            self.fixed_data = data.copy()
        else:
            self.fixed_data = data

        if self.intercept is True:
            self.fixed_data = np.column_stack((np.ones(self.fixed_data.shape[0]), self.fixed_data))

        if self.design_matrix is None:
            self.design_matrix = self.fixed_data
            self.fixed_dim = self.fixed_data.shape[1]
            self.fixed_incl = True
        else:
            raise ValueError("Fixed covariate effects must be defined first.")

    def smooth(self, data: np.ndarray, n_knots: int=20, degree: int=3, rwk: int=2) -> None:
        """
        Method to define smooth effects via the B-Spline basis.

        Args:
            data (np.ndarray): 1-D np.ndarray that contains a covariate.
            n_knots (int, optional): Number of knots. Defaults to 20.
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
        diff_mat = np.diff(np.eye(smooth_matrix.shape[1]), n=rwk, axis=0)
        pen = np.dot(diff_mat.T, diff_mat)
        self.smooth_pen_mat.append(pen)

        if self.design_matrix is None:
            if self.intercept is True:
                self.fixed_data = np.ones((smooth_matrix.shape[0],1))
                self.fixed_dim = 1
                self.fixed_incl = True
                self.smooth_incl = True
            else:
                self.design_matrix = np.asarray(smooth_matrix, dtype=np.float32)
                self.smooth_incl = True
        else:
            if self.smooth_incl is False:
                self.smooth_incl = True

    def center(self) -> np.ndarray:
        if self.smooth_incl is True and self.fixed_incl is True:
            fixed = np.asarray(self.fixed_data, dtype=np.float32)
            self.design_mat_cent= [fixed]
            self.smooth_pen_mat_cent = []
            self.smooth_dim_cent = []
            self.orth_factor = []

            for i in range(len(self.smooth_data)):
                smooth_mat = self.smooth_data[i]
                smooth_mat = np.asarray(smooth_mat, dtype=np.float32)
                c = np.mean(smooth_mat, axis=0)
                c = np.expand_dims(c, axis=1)
                Q, _ = np.linalg.qr(c, mode="complete")

                Tb = Q[:,1:]
                smooth_mat_new = np.dot(smooth_mat, Tb)
                pen_new = Tb.T @ self.smooth_pen_mat[i] @ Tb

                self.design_mat_cent.append(smooth_mat_new)
                self.smooth_pen_mat_cent.append(pen_new)
                self.smooth_dim_cent.append(self.smooth_dim[i]-1)
                self.orth_factor.append(Tb)

            self.design_matrix = np.concatenate(self.design_mat_cent, axis=1)
        else:
            print("No need to include identifiability constraints.")
            raise ValueError("The model must contain at least one fixed parameter in combination with a smooth effect.")

# https://stats.stackexchange.com/questions/517375/splines-relationship-of-knots-degree-and-degrees-of-freedom
# For the identification constraints (sum to zero constrain) check Generalized Additive Models, An introduction with R, Simon N. Wood, page 175 and 211
