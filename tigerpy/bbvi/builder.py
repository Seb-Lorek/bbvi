"""
Black-Box-Variational-Inference.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.tree_util import tree_flatten, tree_unflatten
from jax.example_libraries import optimizers

import numpy as np

import tensorflow as tf
import tensorflow_probability.substrates.jax.distributions as tfjd
import tensorflow_probability.substrates.numpy.distributions as tfnd

import matplotlib.pyplot as plt

from typing import List, Dict, Tuple

from .variational import Variational

from ..model.model import (
    Model,
    Lpred,
    Param,
    Array,
    Any,
    Distribution)

class Bbvi:
    """
    Inference algorithm.
    """

    def __init__(self, Model: Model, num_samples: int, num_iterations: int, seed: Any) -> None:
        self.Model = Model
        self.tree = Model.tree
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.seed = seed
        self.variational_params = {}
        self.opt_variational_params = {}
        self.ELBO = []

        self.set_variational_params()

    def set_variational_params(self) -> None:
        """
        Method to set the internal variational parameters.

        """
        for kw1, input1 in self.Model.y_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                for kw2, input2 in input1.kwinputs.items():
                    self.variational_params[kw2] = self.init_varparam(input2.dim)
            elif isinstance(input1, Param):
                self.variational_params[kw1] = self.init_varparam(input1.dim)

    def init_varparam(self, dim: Any) -> Dict:
        """
        Method to initialize the internal variational parameters.

        Args:
            dim (Any): Dimension of the parameter.

        Returns:
            Dict: The initial interal variational parameters.
        """

        mu = jnp.zeros(dim)

        lower_tri = jnp.diag(jnp.ones(dim))
        return {"mu": mu, "lower_tri": lower_tri}

    def logprob(self, sample: Dict) -> Array:
        """
        Method to calculate the log-probability with a sample from the variational distribution.

        Args:
            sample (Dict): Samples from the variational distribution in dictionary form.

        Returns:
            Array: Log-probability of the model.
        """

        logprior = []
        params = {}
        response_dist = self.tree["response"]["dist"]

        for kw1, input1 in self.Model.y_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                arrays = []
                bijector = response_dist[kw1]["bijector"]
                input1_function = input1.function is not None

                for kw2, input2 in input1.kwinputs.items():
                    if input2.function is not None:
                        transformed = response_dist[kw1][kw2]["bijector"](sample[kw2])
                    else:
                        transformed = sample[kw2]
                    logprior.append(response_dist[kw1][kw2]["dist"].log_prob(transformed))
                    arrays.append(transformed)

                beta = jnp.concatenate(arrays)

                nu = calc(response_dist[kw1]["design_matrix"], beta)
                if input1_function:
                    params[kw1] = bijector(nu)
                else:
                    params[kw1] = nu

            elif isinstance(input1, Param):
                transformed = response_dist[kw1]["bijector"](sample[kw1]) if input1.function is not None else sample[kw1]
                logprior.append(response_dist[kw1]["dist"].log_prob(transformed))
                params[kw1] = transformed

        logprior = jnp.concatenate(logprior)

        loglik = self.tree["response"]["dist"]["type"](**params).log_prob(self.tree["response"]["value"])

        return jnp.sum(loglik, keepdims=True) + jnp.sum(logprior, keepdims=True)

    def pass_samples(self, samples: Dict) -> Array:
        """
        Method to pass the samples from method lower_bound to the logprob method.

        Args:
            samples (Dict): Samples from the variational distribution for the parameters of the model in a dictionary.

        Returns:
            Array: The log-probabilities of the model for all the samples.
        """

        @jit
        def compute_logprob(i):
            sample = {key: value[i] for key, value in samples.items()}
            return self.logprob(sample=sample)

        logprobs = jax.vmap(compute_logprob)(jnp.arange(self.num_samples))

        return logprobs

    def lower_bound(self, variational_params: Dict) -> Array:
        """
        Method to calculate the negative ELBO (evidence lower bound).

        Args:
            variational_params (Dict): The varational paramters in a nested dictionary where the first key identifies the paramter of the model.

        Returns:
            Array: Negative ELBO.
        """

        arrays = []
        samples = {}

        for kw in variational_params.keys():
            mu, lower_tri = variational_params[kw]["mu"], variational_params[kw]["lower_tri"]

            samples[kw] = tfjd.MultivariateNormalTriL(loc=mu, scale_tril=lower_tri).sample(sample_shape=(self.num_samples,), seed=self.seed)

            e = jnp.mean(tfjd.MultivariateNormalTriL(loc=mu, scale_tril=lower_tri).log_prob(samples[kw]), keepdims=True)
            arrays.append(e)

        means = jnp.concatenate(arrays)

        elbo = jnp.mean(self.pass_samples(samples)) - jnp.sum(means)
        return -elbo

    def run_bbvi(self, step_size: float = 0.01) -> Tuple:
        """
        Method to start the stochastic optimization. The method uses adam.

        Args:
            step_size (float, optional): Step size or sometimes also learning rate of the adam optimizer. Defaults to 0.01.

        Returns:
            Tuple: Last negative ELBO and the optimized variational parameters in a dictionary.
        """
        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
        opt_state = opt_init(self.variational_params)
        arrays = []

        @jit
        def step(step, opt_state):
            value, grads = jax.value_and_grad(self.lower_bound)(get_params(opt_state))
            opt_state = opt_update(step, grads, opt_state)
            return value, opt_state

        for i in range(self.num_iterations):
            value, opt_state = step(i, opt_state)
            arrays.append(-jnp.array([value]))

        self.ELBO = jnp.concatenate(arrays)
        self.variational_params = get_params(opt_state)
        self.set_opt_variational_params()
        return value, self.opt_variational_params

    def set_opt_variational_params(self):
        """
        Method to obtain the variational parameters in terms of the covariance matrix and not the lower cholesky factor.

        """

        for kw1, input1 in self.Model.y_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                for kw2, input2 in input1.kwinputs.items():
                    self.opt_variational_params[kw2] = {
                        "mu": self.variational_params[kw2]["mu"],
                        "cov": calc(self.variational_params[kw2]["lower_tri"], self.variational_params[kw2]["lower_tri"].T)
                    }
            elif isinstance(input1, Param):
                self.opt_variational_params[kw1] = {
                    "mu": self.variational_params[kw1]["mu"],
                    "cov": calc(self.variational_params[kw1]["lower_tri"], self.variational_params[kw1]["lower_tri"].T)
                }

    def plot_elbo(self):
        """
        Method to visualize the progression of the ELBO during the optimization.

        """
        plt.plot(jnp.arange(self.num_iterations), self.ELBO)
        plt.title("Progression of the ELBO")
        plt.xlabel("Iteration")
        plt.ylabel("ELBO")
        plt.show()

@jit
def calc(x: Array, y: Array) -> Array:
    """
    Function to calculate matix/dot products.

    Args:
        x (Array): Jax.numpy array. Column dimension must match with row dimension of y.
        y (Array): Jax.numpy array. Row dimension must match with column dimesion of x.

    Returns:
        Array: Jax.numpy array. The matrix/dot product.
    """
    return jnp.dot(x, y)
