"""
Tiger model.
"""

import jax.numpy as jnp

import tensorflow_probability.substrates.jax.distributions as tfjd
import tensorflow_probability.substrates.numpy.distributions as tfnd

from typing import Any, Union

Array = Any
Distribution = Union[tfjd.Distribution, tfnd.Distribution]

from .observation import Obs

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
        self.additional_params = {kw: item for kw, item in self.kwinputs.items()
                             if not isinstance(item, (Hyper, Param, Lpred))} or None

    def init_dist(self) -> Distribution:
        """
        Method to initialize class Dist with kwinputs.value.

        Returns:
            Distribution: A initialized tensorflow probability distribution.
        """

        kwargs = {kw: item.value if isinstance(item, (Hyper, Param, Lpred)) else item for kw, item in self.kwinputs.items()}
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

    def __init__(self, value: Array, distribution: Dist, param_space: Union[str, None] = None, name: str = ""):
        self.value = jnp.atleast_1d(value)
        self.distribution = distribution
        self.param_space = param_space
        self.name = name
        self.dim = self.value.shape[0]
        self.log_prior = self.logprior(value=self.value)

    def logprior(self, value: Array) -> Array:
        """
        Method to calculate the log-probability of the class Param.

        Args:
            value (Array): Current values of the class Param.

        Returns:
            Array: A log-probability.
        """

        log_prior = self.distribution.init_dist().log_prob(value)

        return log_prior

class Lpred:
    """
    Linear predictor.
    """

    def __init__(self, obs: Obs, function: Any = None, **kwinputs: Any):
        self.obs = obs
        self.function = function
        self.kwinputs = kwinputs
        self.design_matrix = jnp.asarray(self.obs.design_matrix, dtype=jnp.float32)
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
        for key, item in self.kwinputs.items():
            arrays.append(item.value)

        params = jnp.concatenate(arrays, dtype=jnp.float32)

        return params

    def update_lpred(self) -> Array:
        """
        Method to update the attribute value by calculating the linear predictor
        and potentially applied an inverse link function (bijetor).

        Returns:
            Array: Value of the linear predictor.
        """

        nu = jnp.dot(self.design_matrix, self.params_values)

        if self.function is not None:
            m = self.function(nu)
        else:
            m = nu

        return m

class Model:
    """
    Static model.
    """

    def __init__(self, response: Array, distribution: Dist):
        self.response = jnp.asarray(response, dtype=jnp.float32)
        self.response_dist = distribution
        self.log_lik = self.loglik()
        self.log_prior = self.logprior()
        self.log_prob = self.logprob()
        self.num_param = self.count_param_instances(self.response_dist)
        self.num_obs = self.response.shape[0]

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
        Compute the log-prior of the model, i.e. the log-pdf of the priors.

        Returns:
            Array: Log-prior of the model.
        """

        prior_list = self.return_param_logpriors(self.response_dist)

        log_prior = jnp.concatenate(prior_list)

        return jnp.sum(log_prior)

    def logprob(self) -> Array:
        """
        Method to calculate the log-probability of the model, i.e. the sum of log-likelihood and log-prior.

        Returns:
            Array: Log-probability of the model.
        """

        return jnp.sum(self.log_lik) + jnp.sum(self.log_prior)

    def return_param_logpriors(self, obj: Any) -> list:
        """
        Method to obtain all log-probabilites of the classes Param. The method has a
        recursive structure.

        Args:
            obj (Any): Class distribution of the response (nested class structure).

        Returns:
            list: Return a list with all log-probabilities.
        """

        log_priors = []

        if isinstance(obj, Param):
            log_priors.append(obj.log_prior)

        elif isinstance(obj, (list, tuple)):
            for item in obj:
                log_priors.extend(self.return_param_logpriors(item))

        elif isinstance(obj, dict):
            for value in obj.values():
                log_priors.extend(self.return_param_logpriors(value))

        elif hasattr(obj, "__dict__"):
            log_priors.extend(self.return_param_logpriors(obj.__dict__))

        return log_priors

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
            count += obj.dim

        if isinstance(obj, (list, tuple)):
            for item in obj:
               count += self.count_param_instances(item)

        elif isinstance(obj, dict):
            for value in obj.values():
                count += self.count_param_instances(value)

        elif hasattr(obj, "__dict__"):
            count += self.count_param_instances(obj.__dict__)

        return count
