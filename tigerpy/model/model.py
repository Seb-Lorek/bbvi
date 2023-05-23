"""
Model container.
"""

import jax.numpy as jnp
from jax import jit
import numpy as np
from typing import Any, Union

import tensorflow_probability.substrates.jax.distributions as tfjd
import tensorflow_probability.substrates.numpy.distributions as tfnd

Array = Any
Distribution = Union[tfjd.Distribution, tfnd.Distribution]

from .observation import Obs

class Model:
    """
    A static model.
    """

    def __init__(self, y: Array, distribution: Distribution) -> None:
        self.y = jnp.asarray(y, dtype=jnp.float32)
        self.y_dist = distribution
        self.log_lik = self.loglik()
        self.log_prior = self.logprior()
        self.log_prob = self.logprob()
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

        # obtain log-probs of the model
        x = jnp.array([], dtype=jnp.float32)
        for kw1, input1 in self.y_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                for kw2, input2 in input1.kwinputs.items():
                    x = jnp.append(x, input2.log_prob)
            elif isinstance(input1, Param):
                x = jnp.append(x, input1.log_prob)

        # sum all the log-priors of the model
        self.log_prior = jnp.sum(x)

        return self.log_prior

    # sum log-likelihood and log-prior i.e. the joint log-probability of the model.
    def logprob(self) -> Array:
        return jnp.sum(self.log_lik) + self.log_prior

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
        self.value = np.asarray(value, dtype=jnp.float32)
        self.distribution = distribution
        self.function = function
        self.name = name
        self.log_prob = self.distribution.init_dist().log_prob(self.value)

class Lpred:
    """
    Class to define a linear predictor.
    """

    def __init__(self, X: Obs, function: Any = None, name: str = "", **kwinputs) -> None:
        self.X = X
        self.function = function
        self.name = name
        self.kwinputs = kwinputs
        self.X_matrix = jnp.asarray(self.X.X, dtype=jnp.float32)
        self.param_value = self.update_params()
        self.value = self.update_lpred()

    def update_params(self):
        x = jnp.array([], dtype=jnp.float32)
        for kw, input in self.kwinputs.items():
            x = jnp.append(x, input.value)
        return x

    def update_lpred(self):
        nu = calc(self.X_matrix, self.param_value)
        if self.function is not None:
            m = self.function(nu)
            return m
        else:
            return nu

# calculate dot products
@jit
def calc(X: Array, param: Array) -> Array:
    return jnp.dot(X, param)
