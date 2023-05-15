"""
Model container
"""

import numpy as np
import inspect
from scipy.interpolate import BSpline as bs
from typing import Any, Union

import tensorflow_probability.substrates.jax.distributions as tfjd
import tensorflow_probability.substrates.numpy.distributions as tfnd

Array = Any
Dist = Union[tfjd.Distribution, tfnd.Distribution]


class Model:
    """
    A static model.
    """
    def __init__(self, intercept=True, X_copy=True):
        self.intercept = intercept
        self.X_copy = X_copy
        self.loglik = None
        self.logprior = None
        self.residuals = None

    # define the response
    def response(self, y: Array, distribution: Dist, invlink: Any = None) -> None:
        """
        Method to store the response vector.

        Args:
            y (Array): A array containing the response vector.
            distribution (Dist): A tensorflow probability distribution that defines the response distribution.
            invlink (Any, optional): Inverse link function that conects the linear predictor to the expectation of the response. Default (None) is the identity link.
        """

        y = np.asarray(y)
        self.y = y
        self.y_dist = distribution
        self.link = invlink

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

        if self.intercept:
            X = np.column_stack((np.ones(len(X)), X))

        if self.X_copy:
            self.X_fixed = X.copy()
        else:
            self.X_fixed = X

    # define smooth effects
    def smooth(self, x: Array, n_knots = 40, degree = 3, rwk = 2) -> None:
        """
        Method to define smooth B-spline covariates

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

    # define the parameters
    def param(self, value: Array, distribution: Dist, name: str, group: str) -> None:
        """
        Method to define the parameters.

        Args:
            value (Array): Inital values for the coefficients.
            distribution (Dist): A object of class Dist.
            name(str): Name of the paramter of the response that you want to model. Must be an attribute of the tensorflow probability choosen for the response.
            type (str): String that defines the group of the parameter either "fixed" or "smooth".
        """
        if type(value) is not np.ndarray:
            value = np.array(value)

        if name in inspect.getfullargspec(self.y_dist).args:
            setattr(self, name + "_" + group + "_" + "value", value)
            setattr(self, name + "_" + group + "_" + "distr", distribution)
        else:
            print("Error: Please provide a paramter of the response distribution.")

    # compute log-likelihood
    def log_lik(self) -> Array:
        """
        Module for the log-likelihood of the model.
        Defined as the sum of the log-probabilities of all observed variables
        with a probability distribution.

        Returns:
            Array: The log-likelihood of the model
        """

        # define here the log-likelihood of the model
        self.loglik = 0 + self.log_prior()

        return self.loglik

    # compute log-prior
    def log_prior(self) -> Array:
        """
        The log-prior of the model.

        Defined as the sum of the log-probabilities of all parameter variables
        with a probability distribution.

        Returns:
            Array: The log-prior of the model.
        """

        # define the log-prior of the model
        self.logprior = 0

        return self.logprior

    # define hyperparameters currently no latent variables except model coefficients
    class Var:
        """
        Hyperparameters.
        """
        def __init__(self, value: Array, name: str = "") -> None:
            self.value = value
            self.name = name

        def __repr__(self) -> str:
            return f'{type(self).__name__}(name="{self.name}")'

    # class to set the distributions of the priors
    class Dist:
        """
        Distribution of the priors
        """

        def __init__(self, distribution: Dist, name: str = "", *inputs: Any, **kwinputs: Any):
            self.distribution = distribution
            self.name = name
            self.inputs = inputs
            self.kwinputs = kwinputs

        def init_dist(self):
            args = [input.value for input in self.inputs]
            kwargs = {kw: input.value for kw, input in self.kwinputs.items()}
            dist = self.distribution(*args, **kwargs)
            return dist
