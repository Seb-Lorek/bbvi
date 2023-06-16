"""
Black-Box-Variational-Inference.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.example_libraries import optimizers

import numpy as np

import tensorflow as tf
import tensorflow_probability.substrates.jax.distributions as tfjd
import tensorflow_probability.substrates.numpy.distributions as tfnd

import matplotlib.pyplot as plt

from typing import List, Dict, Tuple

from .variational import Variational

from ..model.model import (
    Array,
    Any
    )

from ..model.nodes import(
    ModelGraph
)

from..model.utils import (
    dot
)


class Bbvi:
    """
    Inference algorithm.
    """

    def __init__(self, Graph: ModelGraph, num_samples: int, num_iterations: int, seed: Any) -> None:
        self.Graph = Graph
        self.DiGraph = Graph.DiGraph
        self.Model = Graph.Model
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.seed = seed
        self.variational_params = {}
        self.opt_variational_params = {}
        self.elbo_history = {"elbo": jnp.array([])}

        self.set_variational_params()

    def set_variational_params(self) -> None:
        """
        Method to set the internal variational parameters.
        """

        for node in self.Graph.prob_traversal_order:
            node_type = self.DiGraph.nodes[node].get("node_type")
            if node_type == "strong":
                attr = self.DiGraph.nodes[node]["attr"]
                self.variational_params[node] = self.init_varparam(attr["dim"])

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

    def pass_samples(self, samples: Dict) -> Array:
        """
        Method to pass the samples from method lower_bound to the logprob method.

        Args:
            samples (Dict): Samples from the variational distribution for the parameters of the model in a dictionary.

        Returns:
            Array: The log-probabilities of the model for all the samples.
        """

        def compute_logprob(i):
            sample = {key: value[i] for key, value in samples.items()}
            self.Graph.update_graph(sample)
            return self.Graph.logprob()

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

            samples[kw] = tfjd.MultivariateNormalTriL(loc=mu, scale_tril=lower_tri).sample(sample_shape=(self.num_samples), seed=self.seed)

            entropy = jnp.mean(tfjd.MultivariateNormalTriL(loc=mu, scale_tril=lower_tri).log_prob(samples[kw]), keepdims=True)
            arrays.append(entropy)

        means = jnp.concatenate(arrays)

        elbo = jnp.mean(self.pass_samples(samples)) - jnp.sum(means)
        return -elbo

    def run_bbvi(self, step_size: float = 0.001, threshold: float = 0.001) -> Tuple:
        """
        Method to start the stochastic optimization. The method uses adam.

        Args:
            step_size (float, optional): Step size or sometimes also learning rate of the adam optimizer. Defaults to 0.01.

        Returns:
            Tuple: Last negative ELBO and the optimized variational parameters in a dictionary.
        """

        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
        opt_state = opt_init(self.variational_params)
        elbo_history_list = []

        @jit
        def step(step, opt_state):
            value, grads = jax.value_and_grad(self.lower_bound)(get_params(opt_state))
            opt_state = opt_update(step, grads, opt_state)
            return value, opt_state

        for i in range(self.num_iterations):
            value, opt_state = step(i, opt_state)
            elbo_history_list.append(-value)

            if i>1:
                elbo_delta = abs(elbo_history_list[-1] - elbo_history_list[-2])

                if elbo_delta < threshold:
                    break

        self.elbo_history["elbo"] = jnp.array(elbo_history_list)
        self.elbo_history["iter"] = jnp.arange(i + 1)
        self.variational_params = get_params(opt_state)
        self.set_opt_variational_params()
        return value, self.opt_variational_params

    def set_opt_variational_params(self):
        """
        Method to obtain the variational parameters in terms of the covariance matrix and not the lower cholesky factor.
        """

        for node in self.Graph.prob_traversal_order:
            node_type = self.DiGraph.nodes[node].get("node_type")
            if node_type == "strong":
                self.opt_variational_params[node] = {
                        "mu": self.variational_params[node]["mu"],
                        "cov": dot(self.variational_params[node]["lower_tri"], self.variational_params[node]["lower_tri"].T)
                    }

    def plot_elbo(self):
        """
        Method to visualize the progression of the ELBO during the optimization.
        """

        plt.plot(self.elbo_history["iter"], self.elbo_history["elbo"])
        plt.title("Progression of the ELBO")
        plt.xlabel("Iteration")
        plt.ylabel("ELBO")
        plt.show()
