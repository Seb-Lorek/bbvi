"""
Model matrices.
"""

import numpy as np
from scipy.interpolate import BSpline as bs

from .model import (
    Array,
    Any)

class Obs:
    """
    Observations.
    """
    def __init__(self, name: str) -> None:
        self.design_matrix = None
        self.data = None
        self.fixed_data = None
        self.name = name

    def fixed(self, data: Array, intercept: bool = True, data_copy: bool = True) -> None:
        """
        Method to define the fixed covariates.

        Args:
            data (Array): Array that contains the fixed covariates (excluding the intercept).
            intercept (bool, optional): Should the model contain a intercept. Defaults to True.
            data_copy (bool, optional): Should the array be copied. Defaults to True.
        """

        self.intercept = intercept
        self.data_copy = data_copy
        self.fixed_data = True

        if type(data) is not np.array:
            data = np.asarray(data, dtype=np.float32)

        if self.data_copy:
            self.data = data.copy()
        else:
            self.data = data

        if self.intercept:
            self.data = np.column_stack((np.ones(len(self.data)), self.data))

        if self.design_matrix is None:
            self.design_matrix = self.data
        else:
            self.design_matrix = np.column_stack((self.data, self.design_matrix))

    def smooth(self, data: Array, n_knots = 30, degree = 3) -> None:
        """
        Method to define smooth B-spline covariates.

        Args:
            data (Array): 1D-Array that contains a covariate.
            n_knots (int, optional): Number of Knots . Defaults to 30.
            degree (int, optional): The degree of the B-spline. Defaults to 3.
            rwk (int, optional): Random walk order that defines the penalisation of the coefficients. Defaults to 2.
        """

        # first don't include an intercept with smooth effects
        # maybe include later
        self.intercept = False

        knots = np.linspace(data.min(), data.max(), num=n_knots)

        self.data_smooth = bs.design_matrix(data, t=knots, k=degree, extrapolate=True).toarray()
        self.knots_smooth = knots
        self.smooth_dim = self.data_smooth.shape[1]

        if self.design_matrix is None:
            self.design_matrix = self.data_smooth
        else:
            self.design_matrix = np.column_stack((self.design_matrix, self.data_smooth))
