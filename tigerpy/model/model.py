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

    # compute log-likelihood
    def loglik(self) -> Array:
        """
        Module for the log-likelihood of the model.
        Defined as the sum of the log-probabilities of all observed variables
        with a probability distribution.

        Returns:
            Array: The log-likelihood of the model.
        """

        log_lik = self.y_dist.init_dist().log_prob(self.y)

        return log_lik

    # compute log-prior
    def logprior(self) -> Array:
        """
        The log-prior of the model.

        Defined as the sum of the log-probabilities of all parameter variables
        with a probability distribution.

        Returns:
            Array: The log-prior of the model.
        """

        arrays = []
        for kw1, input1 in self.y_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                for kw2, input2 in input1.kwinputs.items():
                    arrays.append(input2.log_prob)
            elif isinstance(input1, Param):
               arrays.append(input1.log_prob)

        # flatten the arrays
        flat_arrays, _ = tree_flatten(arrays)

        # concatenate the flattened arrays
        x = jnp.concatenate(flat_arrays)

        # sum all the log-priors of the model and return sum
        return jnp.sum(x)

    # sum log-likelihood and log-prior i.e. the joint log-probability of the model.
    def logprob(self) -> Array:
        return jnp.sum(self.log_lik) + self.log_prior

    # count the number of parameters in the model
    def countparam(self) -> int:
        count = 0
        for kw1, input1 in self.y_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                for kw2, input2 in input1.kwinputs.items():
                    count += input2.dim[0]
            elif isinstance(input1, Param):
                count += input1.dim[0]
        return count

    # update the logprob with Monte Carlo sample of the variational distribution
    def update_graph(self, sample: dict) -> Array:
        for kw1, input1 in self.y_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                for kw2, input2 in input1.kwinputs.items():
                    input2.value = sample[kw2]
                    input2.log_prob = input2.logprob(value=sample[kw2])
                input1.param_value = input1.update_params()
                input1.value = input1.update_lpred()
            elif isinstance(input1, Param):
                input1.value = sample[kw1]
                input1.log_prob = input1.logprob(value=sample[kw1])

        self.log_lik = self.loglik()
        self.log_prior = self.logprior()
        self.log_prob = self.logprob()

        return self.log_prob

    #build the model tree
    def build_tree(self):
        self.tree["y"] = self.y
        self.tree["y_dist"] = {"dist": self.y_dist.get_dist()}

        for kw1, input1 in self.y_dist.kwinputs.items():
            self.tree["y_dist"][kw1] = {}
            if isinstance(input1, Lpred):
                self.tree["y_dist"][kw1]["fixed"] = input1.X_matrix
                self.tree["y_dist"][kw1]["bijector"] = input1.function
                for kw2, input2 in input1.kwinputs.items():
                    self.tree["y_dist"][kw1][kw2] = {}
                    self.tree["y_dist"][kw1][kw2]["dim"] = input2.dim
                    if input2.function is not None:
                        self.tree["y_dist"][kw1][kw2]["bijector"] = input2.function
                    self.tree["y_dist"][kw1][kw2]["dist"] = input2.distribution.init_dist()
            elif isinstance(input1, Param):
                self.tree["y_dist"][kw1]["dim"] = input1.dim
                if input1.function is not None:
                    self.tree["y_dist"][kw1]["bijector"] = input1.function
                self.tree["y_dist"][kw1]["dist"] = input1.distribution.init_dist()

# define hyperparameters currently no latent variables except model coefficients
class Hyper:
    """
    Hyperparameter.
    """

    def __init__(self, value: Array, name: str = "") -> None:
        self.value = value
        self.name = name

    def __repr__(self) -> str:
        return f'{type(self).__name__}(name="{self.name}")'

# class to set the distributions of the parameters (priors)
class Dist:
    """
    Distribution of a parameter.
    """

    def __init__(self, distribution: Distribution, name: str = "", **kwinputs: Any):
        self.distribution = distribution
        self.name = name
        self.kwinputs = kwinputs
        self.dist_init = self.init_dist()

    # initialize the distribution with kwinputs
    def init_dist(self) -> Distribution:

        kwargs = {kw: input.value for kw, input in self.kwinputs.items()}
        dist = self.distribution(**kwargs)
        return dist

    # get the distribution from class Dist
    def get_dist(self):
        return self.distribution

# class to initialize the parameters
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

    # update the logprob if values have changed
    def logprob(self, value: Array) -> Array:
        return self.distribution.init_dist().log_prob(value)

class Lpred:
    """
    Linear predictor.
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
        arrays = []
        for kw, input in self.kwinputs.items():
            arrays.append(input.value)

        # flatten the arrays
        flat_arrays, _ = tree_flatten(arrays)

        # concatenate the flattened arrays
        x = jnp.concatenate(flat_arrays, dtype=jnp.float32)

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
