"""
Model container.
"""

import numpy as np
import jax

from typing import Any, Union

import tensorflow_probability.substrates.jax.distributions as tfjd
import tensorflow_probability.substrates.numpy.distributions as tfnd

from .observation import Obs

Array = Any
Distribution = Union[tfjd.Distribution, tfnd.Distribution]

class Model:
    """
    A static model.
    """

    def __init__(self, y: Array, distribution: Distribution) -> None:
        self.y = np.asarray(y, dtype=np.float32)
        self.y_dist = distribution
        self.log_lik = self.loglik()
        self.log_prior = None
        self.log_prob = None
        self.residuals = None

    # compute log-likelihood
    def loglik(self) -> Array:
        """
        Module for the log-likelihood of the model.
        Defined as the sum of the log-probabilities of all observed variables
        with a probability distribution.

        Returns:
            Array: The log-likelihood of the model.
        """

        self.log_lik = self.y_dist.init_dist().log_prob(self.y)

        return self.log_lik

    # compute log-prior
    def logprior(self) -> Array:
        """
        The log-prior of the model.

        Defined as the sum of the log-probabilities of all parameter variables
        with a probability distribution.

        Returns:
            Array: The log-prior of the model.
        """

        # define the log-prior of the model
        self.log_prior = 0
        # self.log_prior = self.y_dist.

        return self.log_prior

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

# class to set the distributions of the variables (priors)
class Dist:
    """
    Distribution of the variables.
    """

    def __init__(self, distribution: Distribution, name: str = "", *inputs: Any, **kwinputs: Any):
        self.distribution = distribution
        self.name = name
        self.inputs = inputs
        self.kwinputs = kwinputs

    def init_dist(self) -> Distribution:
        """
        Initialize the distribution of the parameter.

        Returns:
            Dist: A tensorflow probability.
        """

        args = [input.value for input in self.inputs]
        kwargs = {kw: input.value for kw, input in self.kwinputs.items()}
        dist = self.distribution(*args, **kwargs)
        return dist

class Param:
    """
    Class to define parameters and initial values and a distribition.
    """

    def __init__(self, value: Array, distribution: Distribution, function: Any = None, name: str = "") -> None:
        self.value = np.asarray(value, dtype=np.float32)
        self.distribution = distribution
        self.function = function
        self.name = name
        self.log_prob = None

class Lpred:
    """
    Class to define a linear predictor.
    """

    def __init__(self, X: Obs, function: Any = None, name: str = "", **kwinputs) -> None:
        self.X = X
        self.function = function
        self.name = name
        self.kwinputs = kwinputs
        self.value = self.update()

    def update(self):
        param = np.array([])
        for kw, input in self.kwinputs.items():
            param = np.append(param, input.value)
        nu = jax.numpy.dot(self.X.X, param)
        if self.function is not None:
            m = self.function(nu)
            return m
        else:
            return nu
