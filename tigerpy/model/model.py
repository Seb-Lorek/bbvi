"""
Tiger model.
"""

import jax.numpy as jnp

import numpy as np

import tensorflow_probability.substrates.jax.distributions as tfjd
import tensorflow_probability.substrates.numpy.distributions as tfnd

from typing import Any, Union

Array = Any
Distribution = Union[tfjd.Distribution, tfnd.Distribution]

from .observation import Obs

from .utils import (
    dot
)

class Model:
    """
    Static model.
    """

    def __init__(self, response: Array, distribution: Distribution) -> None:
        self.response = jnp.asarray(response, dtype=jnp.float32)
        self.response_dist = distribution
        self.log_lik = self.loglik()
        self.log_prior = self.logprior()
        self.log_prob = self.logprob()
        self.num_param = self.countparam()
        self.tree = {}

    def loglik(self) -> Array:
        """
        Method to calculate the log-likelihood of the model.

        Returns:
            Array: Return the log-likelihood of the model.
        """

        log_lik = self.response_dist.init_dist().log_prob(self.response)

        return log_lik

    def logprior(self) -> Array:
        """
        Compute the log-prior of the model, i.e. the log-pdf/pmf of the priors.

        Returns:
            Array: Log-prior of the model.
        """

        arrays = []
        for kw1, input1 in self.response_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                for kw2, input2 in input1.kwinputs.items():
                    arrays.append(input2.log_prob)
            elif isinstance(input1, Param):
               arrays.append(input1.log_prob)

        # concatenate the flattened arrays
        logprior = jnp.concatenate(arrays)

        return jnp.sum(logprior)

    def logprob(self) -> Array:
        """
        Method to calculate the log-probability of the model, i.e. the sum of log-likelihood and log-prior.

        Returns:
            Array: Log-probability of the model.
        """

        return jnp.sum(self.log_lik) + self.log_prior

    def countparam(self) -> int:
        """
        Method to calculate the number of parameters in the model, being the sum of all dims of class Param.

        Returns:
            int: Number of parameters in the model.
        """

        count = 0
        for kw1, input1 in self.response_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                for kw2, input2 in input1.kwinputs.items():
                    count += input2.dim[0]
            elif isinstance(input1, Param):
                count += input1.dim[0]
        return count

class Hyper:
    """
    Hyperparameter.
    """

    def __init__(self, value: Array, name: str = "") -> None:
        self.value = value
        self.name = name

    def __repr__(self) -> str:
        return f'{type(self).__name__}(name="{self.name}")'

class Dist:
    """
    Distribution.
    """

    def __init__(self, distribution: Distribution, **kwinputs: Any):
        self.distribution = distribution
        self.kwinputs = kwinputs

    def init_dist(self) -> Distribution:
        """
        Method to initialize class Dist with kwinputs.value.

        Returns:
            Distribution: A initialized tensorflow probability distribution.
        """

        kwargs = {kw: input.value for kw, input in self.kwinputs.items()}
        dist = self.distribution(**kwargs)
        return dist


    def get_dist(self) -> Distribution:
        """
        Method to get the tensorflow probability distribution of the class Dist.

        Returns:
            Distribution: A tensorflow probability distribution.
        """

        return self.distribution

class Param:
    """
    Parameter.
    """

    def __init__(self, value: Array, distribution: Dist, function: Any = None, name: str = "") -> None:
        self.value = jnp.atleast_1d(value)
        self.dim = self.value.shape
        self.distribution = distribution
        self.function = function
        self.name = name
        self.log_prob = self.logprob(value=self.value)

    def logprob(self, value: Array) -> Array:
        """
        Method to calculate the log-probability of the class Param.

        Args:
            value (Array): Current values of the class Param.

        Returns:
            Array: A log-probability.
        """

        return jnp.sum(self.distribution.init_dist().log_prob(value), keepdims=True)

class Lpred:
    """
    Linear predictor.
    """

    def __init__(self, Obs: Obs, function: Any = None, **kwinputs: Any) -> None:
        self.Obs = Obs
        self.function = function
        self.kwinputs = kwinputs
        self.design_matrix = jnp.asarray(self.Obs.design_matrix, dtype=jnp.float32)
        self.params_values = self.update_params()
        self.value = self.update_lpred()

    def update_params(self) -> Array:
        """
        Method to update attribute params_values by obtaining all values of class Params in class Lpred.

        Returns:
            Array: Array of all values of class Params in class Lpred.
        """

        arrays = []
        for kw, input in self.kwinputs.items():
            arrays.append(input.value)

        x = jnp.concatenate(arrays, dtype=jnp.float32)

        return x

    def update_lpred(self) -> Array:
        """
        Method to update the attribute value by calculating the linear predictor and potentially applied an inverse link function (bijetor).

        Returns:
            Array: Value of the linear predictor.
        """

        nu = dot(self.design_matrix, self.params_values)

        if self.function is not None:
            m = self.function(nu)
        else:
            m = nu

        return m
