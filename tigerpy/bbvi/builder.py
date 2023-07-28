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

from typing import (
    List,
    Dict,
    Tuple,
    NamedTuple,
    Callable
    )

from ..model.model import (
    Array,
    Any
    )

from ..model.nodes import(
    ModelGraph
)

class BbviState(NamedTuple):
    seed: jax.random.PRNGKey
    opt_state: Any

class Bbvi:
    """
    Inference algorithm.
    """

    def __init__(self, Graph: ModelGraph, key: int, batch_size: int = 256, num_samples: int = 64, num_iterations: int = 5000) -> None:
        self.Graph = Graph
        self.DiGraph = Graph.DiGraph
        self.Model = Graph.Model
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.num_obs = self.Model.response.shape[0]
        self.key = key
        self.variational_params = {}
        self.opt_variational_params = {}
        self.elbo_history = {}

        self.set_variational_params()

    # Still work in progress
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

    # Still work in progress
    def init_variational_params_test(self, joint_nodes: str):
        nodes = joint_nodes.split("_")
        mu = []
        diag = []
        for node in nodes:
            attr = self.DiGraph.nodes[node]["attr"]
            input_pass = self.DiGraph.nodes[node]["input"]
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
                self.variational_params[node] = self.init_variational_params(attr)

    def init_variational_params(self, attr: Any) -> Dict:
        """
        Method to initialize the internal variational parameters.

        Args:
            attr (Any): Attributes of a parameter node.

        Returns:
            Dict: The initial interal variational parameters.
        """

        mu = attr["value"]

        lower_tri = jnp.diag(jnp.ones(attr["dim"]))
        return {"mu": mu, "lower_tri": lower_tri}

    def pass_samples(self, samples: Dict) -> Array:
        """
        Method to pass the samples from method lower_bound to the Graph.logprob() method.

        Args:
            samples (Dict): Samples from the variational distribution for the parameters of the model in a dictionary.

        Returns:
            Array: The log-probabilities of the model for all the samples.
        """

        def compute_logprob(sample):
            self.Graph.update_graph(sample)
            return self.Graph.logprob()

        logprobs = jax.vmap(compute_logprob)(samples)

        return logprobs

    def pass_samples_minibtaching(self, samples: Dict, seed: jax.random.PRNGKey) -> Array:
        """
        Method to pass the samples from method lower_bound to the Graph.logprob() method.

        Args:
            samples (Dict): Samples from the variational distribution for the parameters of the model in a dictionary.

        Returns:
            Array: The log-probabilities of the model for all the samples.
        """

        idx = jax.random.permutation(seed, self.num_obs)

        def compute_logprob(sample):
            arrays = []

            for i in range(self.num_obs // self.batch_size):
                self.Graph.update_graph(sample, idx[i * self.batch_size : (i + 1) * self.batch_size])
                arrays.append(self.Graph.logprob())

            mean = jnp.concatenate(arrays)

            return jnp.mean(mean, keepdims=True)

        logprobs = jax.vmap(compute_logprob)(samples)

        return logprobs

    def lower_bound(self, variational_params: Dict, seed: jax.random.PRNGKey) -> Array:
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
            if self.DiGraph.nodes[kw]["attr"]["param_space"] is None:

                samples[kw] = tfjd.MultivariateNormalTriL(loc=mu, scale_tril=lower_tri).sample(sample_shape=self.num_samples, seed=seed)
                entropy = jnp.mean(tfjd.MultivariateNormalTriL(loc=mu, scale_tril=lower_tri).log_prob(samples[kw]), keepdims=True)
                arrays.append(entropy)

            elif self.DiGraph.nodes[kw]["attr"]["param_space"] == "positive":

                samples[kw] = jnp.exp(tfjd.MultivariateNormalTriL(loc=mu, scale_tril=lower_tri).sample(sample_shape=self.num_samples, seed=seed))
                l = tfjd.MultivariateNormalTriL(loc=mu, scale_tril=lower_tri).log_prob(jnp.log(samples[kw]))
                e = jnp.multiply(l, jnp.squeeze(1/samples[kw]))
                entropy = jnp.mean(l, keepdims=True)
                arrays.append(entropy)

        means = jnp.concatenate(arrays)

        elbo = jnp.mean(self.pass_samples_minibtaching(samples, seed)) - jnp.sum(means)

        return - elbo

    def run_bbvi(self, step_size: float = 0.001, threshold: float = 1e-5, chunk_size: int = 1) -> Tuple:
        """
        Method to start the stochastic optimization. The implementation uses adam.

        Args:
            step_size (float, optional): Step size of the SGD optimizer (Adam). Defaults to 0.001.
            threshold (float, optional): Threshold to stop the optimization. Defaults to 1e-5.
            chunk_size (int, optional): Chunk sizes of to evaluate. Defaults to 1.
            batch_size (int, optional): Size of batches to compute the ELBO. Defaults to 256.

        Returns:
            Tuple: Last ELBO and the optimized variational parameters in a dictionary.
        """

        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
        opt_state = opt_init(self.variational_params)

        def bbvi_body(state, step):
            value, grads = jax.value_and_grad(self.lower_bound)(get_params(state.opt_state), state.seed)
            opt_state = opt_update(step, grads, state.opt_state)

            return BbviState(state.seed, opt_state), -value

        @partial(jax.jit, static_argnums=0)
        def bbvi_scan(chunk_size: int, bbvi_state: BbviState) -> Tuple[BbviState, jnp.array]:

            scan_input = jnp.arange(chunk_size)
            new_state, elbo_chunk = jax.lax.scan(bbvi_body, bbvi_state, scan_input)

            return new_state, elbo_chunk

        seed = jax.random.PRNGKey(self.key)
        state = BbviState(seed, opt_state)
        elbo_chunks = []

        for _ in range(self.num_iterations // chunk_size):
            state, elbo_chunk = bbvi_scan(chunk_size, state)
            elbo_chunks.append(elbo_chunk)
            elbo_delta = abs(elbo_chunk[-1] - elbo_chunk[-2]) if len(elbo_chunk) > 1 else jnp.inf
            if elbo_delta < threshold:
                break

        self.elbo_history["elbo"] = jnp.concatenate(elbo_chunks)
        self.elbo_history["iter"] = jnp.arange(len(self.elbo_history["elbo"]))
        self.variational_params = get_params(state.opt_state)
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
                        "cov": jnp.dot(self.variational_params[node]["lower_tri"], self.variational_params[node]["lower_tri"].T)
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
