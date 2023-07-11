"""
Black-Box-Variational-Inference.
"""

import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

import tensorflow_probability.substrates.jax.distributions as tfjd
import tensorflow_probability.substrates.numpy.distributions as tfnd

import matplotlib.pyplot as plt

import functools
from functools import partial

from typing import List, Dict, Tuple, NamedTuple, Callable

from .variational import Variational

from ..model.model import (
    Array,
    Any
    )

from ..model.nodes import(
    ModelGraph
)

from ..utils import (
    dot
)

class BbviState(NamedTuple):
    seed: jax.random.PRNGKey
    opt_state: Any

class Bbvi:
    """
    Inference algorithm.
    """

    def __init__(self, Graph: ModelGraph, num_samples: int, num_iterations: int, key: Any) -> None:
        self.Graph = Graph
        self.DiGraph = Graph.DiGraph
        self.Model = Graph.Model
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.key = key
        self.variational_params = {}
        self.opt_variational_params = {}
        self.elbo_history = {}

        self.set_variational_params()

    def set_variational_params_test(self) -> None:

        included = []
        for node in reversed(self.Graph.prob_traversal_order):
            node_type = self.DiGraph.nodes[node].get("node_type")
            if node_type == "strong" and node not in included:
                included.append(node)
                neighbours = list(self.DiGraph.predecessors(node))
                neighbours_param = []
                for neighbour in neighbours:
                    neighbour_node_type = self.DiGraph.nodes[neighbour].get("node_type")
                    if neighbour_node_type == "strong":
                        neighbours_param.append(neighbour)

                included.extend(neighbours_param)
                if len(neighbours_param) >= 1:
                    name = node + "_" + "_".join(neighbours_param)
                    self.variational_params[name] = self.init_variational_params_test(name)
                else:
                    self.variational_params[node] = self.init_variational_params(node)

    def init_variational_params_test(self, joint_nodes: str):
        nodes = joint_nodes.split("_")
        mu = []
        diag = []
        for node in nodes:
            attr = self.DiGraph.nodes[node]["attr"]
            input = self.DiGraph.nodes[node]["input"]
            mu.append(attr["internal_value"])
            diag.append(jnp.ones(attr["dim"])*1.25)

        mu = jnp.concatenate(mu)
        lower_tri = jnp.diag(jnp.concatenate(diag))

        return {"mu": mu, "lower_tri": lower_tri}

    def set_variational_params(self) -> None:
        """
        Method to set the internal variational parameters.
        """

        for node in self.Graph.prob_traversal_order:
            node_type = self.DiGraph.nodes[node].get("node_type")
            if node_type == "strong":
                attr = self.DiGraph.nodes[node]["attr"]
                input = self.DiGraph.nodes[node]["input"]
                self.variational_params[node] = self.init_variational_params(attr, input)

    def init_variational_params(self, attr: Any, input: Any) -> Dict:
        """
        Method to initialize the internal variational parameters.

        Args:
            attr (Any): Attributes of a parameter node.
            input (Any): Input data that is passed from child notes to parents.

        Returns:
            Dict: The initial interal variational parameters.
        """

        mu = attr["internal_value"]

        lower_tri = jnp.diag(jnp.ones(attr["dim"])*1.25)
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

            samples[kw] = tfjd.MultivariateNormalTriL(loc=mu, scale_tril=lower_tri).sample(sample_shape=(self.num_samples), seed=jax.random.PRNGKey(self.key))

            entropy = jnp.mean(tfjd.MultivariateNormalTriL(loc=mu, scale_tril=lower_tri).log_prob(samples[kw]), keepdims=True)
            arrays.append(entropy)

        means = jnp.concatenate(arrays)

        elbo = jnp.mean(self.pass_samples(samples)) - jnp.sum(means)
        return -elbo

    @partial(jax.jit, static_argnums=(0,1))
    def bbvi_scan(self, chunk_size: int, bbvi_state: BbviState) -> Tuple[BbviState, jnp.array]:

        def bbvi_body(state, i, lower_bound, get_params, opt_update):
            value, grads = jax.value_and_grad(lower_bound)(get_params(state.opt_state))
            opt_state = opt_update(i, grads, state.opt_state)

            return BbviState(state.seed, opt_state), -value

         # pass self.lower_bound, self.get_params, self.opt_update as arguments to bbvi_body
        bbvi_body = functools.partial(bbvi_body, lower_bound=self.lower_bound, get_params=self.get_params, opt_update=self.opt_update)

        scan_input = jnp.arange(chunk_size)
        new_state, elbo_chunk = jax.lax.scan(bbvi_body, bbvi_state, scan_input)

        return new_state, elbo_chunk

    def run_bbvi(self, step_size: float = 0.001, threshold: float = 1e-5, chunk_size: int = 1) -> Tuple:
        """
        Method to start the stochastic optimization. The implementation ueses adam.

        Args:
            step_size (float, optional): Step size or sometimes also learning rate of the adam optimizer. Defaults to 0.01.

        Returns:
            Tuple: Last negative ELBO and the optimized variational parameters in a dictionary.
        """

        self.opt_init, self.opt_update, self.get_params = optimizers.adam(step_size=step_size)
        opt_state = self.opt_init(self.variational_params)

        seed = jax.random.PRNGKey(self.key)
        state = BbviState(seed, opt_state)
        elbo_chunks = []

        for _ in range(self.num_iterations // chunk_size):
            state, elbo_chunk = self.bbvi_scan(chunk_size, state)
            elbo_chunks.append(elbo_chunk)
            elbo_delta = abs(elbo_chunk[-1] - elbo_chunk[-2]) if len(elbo_chunk) > 1 else jnp.inf
            if elbo_delta < threshold:
                break

        elbo_history = jnp.concatenate(elbo_chunks)
        self.elbo_history["elbo"] = elbo_history
        self.elbo_history["iter"] = jnp.arange(len(elbo_history))
        self.variational_params = self.get_params(state.opt_state)
        self.set_opt_variational_params()
        return self.elbo_history["elbo"][-1], self.opt_variational_params

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
