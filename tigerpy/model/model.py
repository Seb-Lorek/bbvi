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

from ..utils import (
    dot
)

class Model:
    """
    Static model.
    """

    def __init__(self, response: Array, distribution: Distribution):
        self.response = jnp.asarray(response, dtype=jnp.float32)
        self.response_dist = distribution
        self.log_lik = self.loglik()
        self.log_prior = self.logprior()
        self.log_prob = self.logprob()
        self.num_param = self.count_param_instances(self.response_dist)

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

        # Obtain all logprior arrays from the model
        prior_list = self.return_param_logpriors(self.response_dist)

        # concatenate the flattened arrays
        log_prior = jnp.concatenate(prior_list)

        return jnp.sum(log_prior)

    def logprob(self) -> Array:
        """
        Method to calculate the log-probability of the model, i.e. the sum of log-likelihood and log-prior.

        Returns:
            Array: Log-probability of the model.
        """

        return jnp.sum(self.log_lik) + self.log_prior

    def return_param_logpriors(self, obj: Any) -> list:
        """
        Method to obtain all log-probabilites of the classes Param. The function has a
        recursive structure.

        Args:
            obj (Any): Class distribution of the response (nested class structure).

        Returns:
            list: Return a list with all log-probabilites.
        """

        l = []

        if isinstance(obj, Param):
            l.extend([obj.log_prior])

        elif isinstance(obj, (list, tuple)):
            for item in obj:
                l.extend(self.return_param_logpriors(item))

        elif isinstance(obj, dict):
            for value in obj.values():
                l.extend(self.return_param_logpriors(value))

        elif hasattr(obj, "__dict__"):
            l.extend(self.return_param_logpriors(obj.__dict__))

        return l

    def count_param_instances(self, obj: Any) -> int:
        """
        Method to count the number of parameters in the model. The method
        has a recursive structure.

        Args:
            obj (Any): Class distribution of the response (nested class structure).

        Returns:
            int: Number of parameters in the model.
        """

        count = 0
        if isinstance(obj, Param):
            count += obj.dim[0]

        if isinstance(obj, (list, tuple)):
            for item in obj:
               count += self.count_param_instances(item)

        elif isinstance(obj, dict):
            for value in obj.values():
                count += self.count_param_instances(value)

        elif hasattr(obj, "__dict__"):
            count += self.count_param_instances(obj.__dict__)

        return count

class Hyper:
    """
    Hyperparameter.
    """

    def __init__(self, value: Array, name: str = ""):
        self.value = jnp.asarray(value)
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
        self.fixed_params = {kw: input for kw, input in self.kwinputs.items()
                             if not isinstance(input, (Hyper, Param, Lpred))} or None

    def init_dist(self) -> Distribution:
        """
        Method to initialize class Dist with kwinputs.value.

        Returns:
            Distribution: A initialized tensorflow probability distribution.
        """

        kwargs = {kw: input.value if isinstance(input, (Hyper, Param, Lpred)) else input for kw, input in self.kwinputs.items()}
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

    def __init__(self, internal_value: Array, distribution: Dist, function: Any = None, name: str = ""):
        self.distribution = distribution
        self.function = function
        self.name = name
        self.internal_value = jnp.atleast_1d(internal_value)
        self.value = self.init_value(self.internal_value)
        self.dim = self.value.shape
        self.log_prior = self.logprior(value=self.value)

    def init_value(self, interal_value):
        if self.function is not None:
            transform = self.function(interal_value)
        else:
            transform = jnp.atleast_1d(interal_value)
        return transform

    def logprior(self, value: Array) -> Array:
        """
        Method to calculate the log-probability of the class Param.

        Args:
            value (Array): Current values of the class Param.

        Returns:
            Array: A log-probability.
        """

        log_prior = self.distribution.init_dist().log_prob(value)

        return jnp.sum(log_prior, keepdims=True)

class Lpred:
    """
    Linear predictor.
    """

    def __init__(self, Obs: Obs, function: Any = None, **kwinputs: Any):
        self.Obs = Obs
        self.function = function
        self.kwinputs = kwinputs
        self.design_matrix = jnp.asarray(self.Obs.design_matrix, dtype=jnp.float32)
        self.params_values = self.update_params()
        self.value = self.update_lpred()

    def update_params(self) -> Array:
        """
        Method to update attribute params_values by obtaining all values of
        class Params in class Lpred.

        Returns:
            Array: Array of all values of class Params in class Lpred.
        """

        arrays = []
        for kw, input in self.kwinputs.items():
            arrays.append(input.value)

        params = jnp.concatenate(arrays, dtype=jnp.float32)

        return params

    def update_lpred(self) -> Array:
        """
        Method to update the attribute value by calculating the linear predictor
        and potentially applied an inverse link function (bijetor).

        Returns:
            Array: Value of the linear predictor.
        """

        nu = dot(self.design_matrix, self.params_values)

        if self.function is not None:
            m = self.function(nu)
        else:
            m = nu

        return m
