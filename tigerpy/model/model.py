"""
Tiger model.
"""

import jax.numpy as jnp
import jax 

import tensorflow_probability.substrates.jax.distributions as tfjd
import tensorflow_probability.substrates.numpy.distributions as tfnd

from typing import (
    Any, 
    Union,
    Callable
)

Array = Any
Distribution = Union[tfjd.Distribution, tfnd.Distribution]

from .observation import Obs

class Hyper:
    """
    Hyperparameter.
    """

    def __init__(self, value: Array, name: str=""):
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

    def init_dist(self) -> Distribution:
        """
        Method to initialize the probability distribution of class Dist with kwinputs.value 
        or the additional_params.

        Returns:
            Distribution: A initialized tensorflow probability distribution.
        """

        kwargs = {kw: item.value for kw, item in self.kwinputs.items()}
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

    def __init__(self, value: Array, distribution: Dist, param_space: Union[str, None]=None, name: str=""):
        self.value = jnp.atleast_1d(value)
        self.distribution = distribution
        self.param_space = param_space
        self.name = name
        self.dim = self.value.shape[0]
        self.log_prior = self.logprior()

    def logprior(self) -> jax.Array:
        """
        Method to calculate the log-probability of the class Param.

        Args:
            value (Array): Current values of the class Param.

        Returns:
            Array: A log-probability.
        """

        log_prior = self.distribution.init_dist().log_prob(self.value)

        return log_prior

class Lpred:
    """
    Linear predictor.
    """

    def __init__(self, obs: Obs, 
                 function: Union[Callable, None]=None, 
                 name: str ="",
                 **kwinputs: Any):
        self.obs = obs
        self.function = function
        self.name = name
        self.kwinputs = kwinputs
        self.design_matrix = jnp.asarray(self.obs.design_matrix, dtype=jnp.float32)
        self.params_values = self.update_params()
        self.value = self.update_lpred()

    def update_params(self) -> jax.Array:
        """
        Method to update attribute params_values by obtaining all values of
        class Params in class Lpred.

        Returns:
            jax.Array: Array of all values of class Params in class Lpred.
        """

        arrays = []
        for key, item in self.kwinputs.items():
            arrays.append(item.value)

        params = jnp.concatenate(arrays, dtype=jnp.float32)

        return params

    def update_lpred(self) -> jax.Array:
        """
        Method to update the attribute value by calculating the linear predictor
        and potentially apply an inverse link function.

        Returns:
            jax.Array: Values of the linear predictor.
        """

        nu = jnp.dot(self.design_matrix, self.params_values)

        if self.function is not None:
            transformed = self.function(nu)
        else:
            transformed = nu

        return transformed

class Calc:
    """
    Calculations.
    """

    def __init__(self, function: Callable, **kwinputs: Any):
        self.function = function
        self.kwinputs = kwinputs
        self.value = self.calc_value()
    
    def calc_value(self):
        """
        Method to calculate the values of a calculation node. 

        Returns:
            jax.Array: Values of the calculation node.
        """
        kwargs = {kw: item.value for kw, item in self.kwinputs.items()}
        
        return self.function(**kwargs)

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

    def loglik(self) -> jax.Array:
        """
        Method to calculate the log-likelihood of the model.

        Returns:
            Array: Return the log-likelihood of the model.
        """

        log_lik = self.response_dist.init_dist().log_prob(self.response)

        return log_lik

    def logprior(self) -> jax.Array:
        """
        Compute the log-prior of the model, i.e. the log-pdf of all priors in the model.

        Returns:
            jax.Array: Log-prior of the model.
        """
        prior_list, nodes = self.return_param_logpriors(self.response_dist)
        unqiue_nodes = []
        unique_prior_list = []
        for prior, node in zip(prior_list, nodes):
            if node not in unqiue_nodes:
                unique_prior_list.append(prior)
                unqiue_nodes.append(node)

        log_prior = jnp.concatenate(unique_prior_list)
        
        return jnp.sum(log_prior)

    def logprob(self) -> jax.Array:
        """
        Method to calculate the log-probability of the model, i.e. the sum of log-likelihood and log-prior.

        Returns:
            jax.Array: Log-probability of the model.
        """

        return jnp.sum(self.log_lik) + self.log_prior

    def return_param_logpriors(self, obj: Any) -> list:
        """
        Method to obtain all log-probabilites of the classes Param. The method has a
        recursive structure.

        Args:
            obj (Any): Classes of the stacked model (nested class structure).

        Returns:
            list: Return a list with all log-probabilities.
        """

        log_priors = []
        nodes = []
        
        if isinstance(obj, Param):
            if obj.name not in nodes:
                log_priors.append(obj.log_prior)
                nodes.append(obj.name)

        if isinstance(obj, (list, tuple)):
            for item in obj:
                prior, node = self.return_param_logpriors(item)
                log_priors.extend(prior)
                nodes.extend(node)
        elif isinstance(obj, dict):
            for value in obj.values():
                prior, node = self.return_param_logpriors(value)
                log_priors.extend(prior)
                nodes.extend(node)
        elif hasattr(obj, "__dict__"):
            prior, node = self.return_param_logpriors(obj.__dict__)
            log_priors.extend(prior)
            nodes.extend(node)

        return log_priors, nodes

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
