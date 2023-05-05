"""
Model container
"""

import numpy as np

class Model:

    def __init__(self, fit_intercept=True, copy_X=True):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.coef = None
        self.latents = None

    # define the observations
    def Obs(self, data, distribution, name):
        """Module to store the observations.

        Args:
            data (list(), np.array()): The observed data.
            distribution (tfp.distributions.Distribution()): A TFP distribution.
            name (str()): Either "response", "covariates" or "smooth".
        """

        data = np.asarray(data)

        if name == "response":
            self.y = data
            self.y_dist = distribution

        if name == "covariates" and distribution == None:
            X = data

            if self.fit_intercept:
                X = np.column_stack((np.ones(len(X)), X))

            if self.copy_X:
                self._X = X.copy()

        if name == "smooth" and distribution == None:
            X = data

            if self.copy_X:
                self._X = X.copy()


    # define here the parameters


    # define variables/hyperparameters


    # build the model
