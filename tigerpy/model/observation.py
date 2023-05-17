"""
Model matrices.
"""

import numpy as np
from scipy.interpolate import BSpline as bs
from typing import Any

Array = Any

# class to define covariate matrices
class Obs:
    """
    Observations.
    """
    def __init__(self, name: str) -> None:
        self.X = None
        self.name = name

    # define the fixed covariates
    def fixed(self, X: Array, intercept=True, X_copy=True) -> None:
        """
        Method to define the fixed covariates.

        Args:
            X (Array): A array that contains the fixed covariates (excluding the intercept).
            intercept (bool, optional): Should a intercept be included. Defaults to True.
            X_copy (bool, optional): Should the array be copied? Defaults to True.
        """

        # potentially redefine intercept and copy_x
        self.intercept = intercept
        self.X_copy = X_copy

        if type(X) is not np.array:
            X = np.asarray(X, dtype=np.float32)

        if self.intercept:
            X = np.column_stack((np.ones(len(X)), X))

        if self.X_copy:
            self.X_fixed = X.copy()
        else:
            self.X_fixed = X

        if self.X == None:
            self.X = self.X_fixed.copy()
        else:
            self.X = np.column_stack((self.X, self.X_fixed))

    # define smooth effects
    def smooth(self, x: Array, n_knots = 40, degree = 3, rwk = 2) -> None:
        """
        Method to define smooth B-spline covariates.

        Args:
            x (Array): Array that contains the the covariate.
            n_knots (int, optional): Number of Knots . Defaults to 40.
            degree (int, optional): The degree of the B-spline. Defaults to 3.
            rwk (int, optional): Random walk order that defines the penalisation of the coefficients. Defaults to 2.
        """

        # first don't include an intercept with smooth effects
        # maybe include later
        self.intercept = False

        knots = np.linspace(x.min(), x.max(), num=n_knots)

        self.X_smooth = bs.design_matrix(x, t=knots, k=degree, extrapolate=True)
        self.knots_smooth = knots
        self.order_smooth = rwk

        if self.X == None:
            self.X = self.X_smooth
        else:
            self.X = np.column_stack((self.X, self.X_smooth))
