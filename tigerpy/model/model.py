"""
Tiger model.
"""

from jax import jit
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten

import numpy as np

import tensorflow_probability.substrates.jax.distributions as tfjd
import tensorflow_probability.substrates.numpy.distributions as tfnd

from typing import Any, Union

Array = Any
Distribution = Union[tfjd.Distribution, tfnd.Distribution]

from .observation import Obs

class Model:
    """
    Static model.
    """

    def __init__(self, y: Array, distribution: Distribution) -> None:
        self.y = jnp.asarray(y, dtype=jnp.float32)
        self.y_dist = distribution
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

        log_lik = self.y_dist.init_dist().log_prob(self.y)

        return log_lik

    def logprior(self) -> Array:
        """
        Compute the log-prior of the model, i.e. the log-pdf/pmf of the priors.

        Returns:
            Array: Log-prior of the model.
        """

        arrays = []
        for kw1, input1 in self.y_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                for kw2, input2 in input1.kwinputs.items():
                    arrays.append(input2.log_prob)
            elif isinstance(input1, Param):
               arrays.append(input1.log_prob)

        # concatenate the flattened arrays
        logprior = jnp.concatenate(arrays)

        return jnp.sum(logprior, keepdims=True)

    def logprob(self) -> Array:
        """
        Method to calculate the log-probability of the model, i.e. the sum of log-likelihood and log-prior.

        Returns:
            Array: Log-probability of the model.
        """
        return jnp.sum(self.log_lik, keepdims=True) + self.log_prior

    def countparam(self) -> int:
        """
        Method to calculate the number of parameters in the model, being the sum of all dims of class Param.

        Returns:
            int: Number of parameters in the model.
        """
        count = 0
        for kw1, input1 in self.y_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                for kw2, input2 in input1.kwinputs.items():
                    count += input2.dim[0]
            elif isinstance(input1, Param):
                count += input1.dim[0]
        return count


    def update_graph(self, sample: dict) -> Array:
        """
        Method to update the graph with a sample from the variational distribution for all Params.

        Args:
            sample (dict): Samples from the variational distribution in dictionary form.

        Returns:
            Array: Current log-probability of the model.
        """
        for kw1, input1 in self.y_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                for kw2, input2 in input1.kwinputs.items():
                    input2.value = sample[kw2]
                    input2.log_prob = input2.logprob(value=sample[kw2])
                input1.params_values = input1.update_params()
                input1.value = input1.update_lpred()
            elif isinstance(input1, Param):
                input1.value = sample[kw1]
                input1.log_prob = input1.logprob(value=sample[kw1])

        self.log_lik = self.loglik()
        self.log_prior = self.logprior()
        self.log_prob = self.logprob()

        return self.log_prob

    def build_tree(self) -> None:
        """
        Method to build a model tree using a dictionary structure.

        """
        self.tree["response"] = {}
        self.tree["response"]["value"] = self.y
        self.tree["response"]["dist"] = {"type": self.y_dist.get_dist()}

        for kw1, input1 in self.y_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                self.tree["response"]["dist"][kw1] = {
                    "design_matrix": input1.design_matrix,
                    "bijector": input1.function,
                    **{
                        kw2: {
                            "dim": input2.dim,
                            "bijector": input2.function,
                            "dist": input2.distribution.init_dist()
                        }
                        for kw2, input2 in input1.kwinputs.items()
                    }
                }
            elif isinstance(input1, Param):
                self.tree["response"]["dist"][kw1] = {
                    "dim": input1.dim,
                    "bijector": input1.function,
                    "dist": input1.distribution.init_dist()
                }

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
        self.dist_init = self.init_dist()

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

    def __init__(self, value: Array, distribution: Distribution, function: Any = None, name: str = "") -> None:
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

        nu = calc(self.design_matrix, self.params_values)

        if self.function is not None:
            m = self.function(nu)
        else:
            m = nu

        return m

@jit
def calc(design_matrix: Array, params: Array) -> Array:
    """
    Function to calcute a matrix product using jit.

    Args:
        design_matrix (Array): Design matrix of class Obs
        params (Array): Attribute params_values of class Lred

    Returns:
        Array: Matrix product of the design matrix and the parameters in the class Lpred.
    """

    return jnp.dot(design_matrix, params)
