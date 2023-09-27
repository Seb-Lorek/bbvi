"""
Black-Box-Variational-Inference.
"""

import jax
import jax.numpy as jnp

import optax

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
    Callable,
    Union
)

from .transform import (
    log_transform,
    batched_jac_determinant
)

from ..model.model import (
    Array,
    Any
)

from ..model.nodes import (
    ModelGraph
)

from ..distributions.mvn import (
    mvn_precision_chol_log_prob,
    mvn_precision_chol_sample,
    solve_chol
)

class BbviState(NamedTuple):
    key: int
    opt_state: Any
    params: Any

class Bbvi:
    """
    Inference algorithm.
    """

    def __init__(self, graph: ModelGraph) -> None:
        self.graph = graph
        self.digraph = graph.digraph
        self.model = graph.model
        self.num_obs = self.model.num_obs
        self.init_variational_params = {}
        self.variational_params = {}
        self.opt_variational_params = {}
        self.elbo_hist = {}

        self.set_variational_params()

    def set_variational_params(self) -> None:
        """
        Method to set the internal variational parameters.
        """

        for node in self.graph.prob_traversal_order:
            node_type = self.digraph.nodes[node]["node_type"]
            if node_type == "strong":
                attr = self.digraph.nodes[node]["attr"]
                self.init_variational_params[node] = self.starting_variational_params(attr)

    def starting_variational_params(self, attr: Any) -> Dict:
        """
        Method to initialize the variational parameters.

        Args:
            attr (Any): Attributes of a parameter node.

        Returns:
            Dict: The initial interal variational parameters.
        """

        # Setting start values to attr["value"] or jnp.log(attr["value"])
        if attr["param_space"] is None:
            loc = attr["value"]
            lower_tri = jnp.diag(jnp.ones(attr["dim"]))
        elif attr["param_space"] == "positive":
            loc = jnp.log(attr["value"])
            lower_tri = jnp.diag(jnp.ones(attr["dim"]))

        return {"loc": loc, "lower_tri": lower_tri}

    def mc_logprob(self, samples: Dict, batch_idx: Array, num_var_samples: int) -> Array:
        """
        Method to pass the samples from method lower_bound to the graph.logprob() method.

        Args:
            samples (Dict): Samples from the variational distribution for the parameters of the model in a dictionary.
            batch_idx (Array): Indexes of the mini-batch.
            num_var_samples(int): Number of samples from the variational distribution.

        Returns:
            Array: Monte carlo integral for the noisy log-probability of the model. We only use a subsample
            of the data
        """

        # update the DAG
        self.graph.update_graph(samples, batch_idx)

        # calculate the scaled log-likelihood
        log_lik = self.digraph.nodes["response"]["attr"]["log_lik"]
        scaled_log_lik = self.num_obs * jnp.mean(log_lik, axis=-1)
        # print("Log-lik", scaled_log_lik.shape)

        log_prior = self.graph.collect_logpriors(num_var_samples)
        # print("Log-prior:", log_prior.shape)

        return jnp.mean(scaled_log_lik + log_prior)

    # Should also take the sample_size
    def lower_bound(self,
                    variational_params: Dict,
                    batch_idx: Array,
                    num_var_samples: int,
                    key: jax.random.PRNGKey) -> Array:
        """
        Method to calculate the negative ELBO (evidence lower bound).

        Args:
            variational_params (Dict): The varational paramters in a nested dictionary where the first key identifies the paramter of the model.

        Returns:
            Array: Negative ELBO.
        """

        key, *subkeys = jax.random.split(key, len(variational_params)+1)

        arrays = []
        samples = {}

        i = 0
        for kw in variational_params.keys():
            loc, lower_tri = variational_params[kw]["loc"], variational_params[kw]["lower_tri"]
            if self.digraph.nodes[kw]["attr"]["param_space"] is None:
                lower_tri = jnp.tril(lower_tri)

                samples[kw] = mvn_precision_chol_sample(loc=loc, precision_matrix_chol=lower_tri, key=subkeys[i], S=num_var_samples)
                l = mvn_precision_chol_log_prob(x=samples[kw], loc=loc, precision_matrix_chol=lower_tri)

                entropy = jnp.mean(l, keepdims=True)

                arrays.append(entropy)

            elif self.digraph.nodes[kw]["attr"]["param_space"] == "positive":
                lower_tri = jnp.tril(lower_tri)

                s = mvn_precision_chol_sample(loc=loc, precision_matrix_chol=lower_tri, key=subkeys[i], S=num_var_samples)
                l = mvn_precision_chol_log_prob(x=s, loc=loc, precision_matrix_chol=lower_tri)

                samples[kw] = jnp.exp(s)

                # Adjust the density of the variational distribution to account for the parameter space transformation
                test = batched_jac_determinant(log_transform, samples[kw])
                l_adjust = l + jnp.log(test)
                entropy = jnp.mean(l_adjust, keepdims=True)

                arrays.append(entropy)
            i += 1

        means = jnp.concatenate(arrays)

        elbo = self.mc_logprob(samples, batch_idx, num_var_samples) - jnp.sum(means)

        return - elbo

    def run_bbvi(self,
                 step_size: Union[Any, float] = 0.01,
                 threshold: float = 1e-2,
                 key_int: int = 1,
                 batch_size: int = 32,
                 num_var_samples: int = 64,
                 chunk_size: int = 1,
                 epochs: int = 1000) -> tuple:
        """
        Method to start the stochastic gradient optimization. The implementation uses Adam.

        Args:
            step_size (float, optional): Step size (learning rate) of the SGD optimizer (Adam).
            Can also be a scheduler. Defaults to 0.001.
            threshold (float, optional): Threshold to stop the optimization. Defaults to 1e-5.
            batch_size (int, optional): Batchsize, i.e. number of samples to use during SGD. Defaults to 32.
            num_var_samples (int, optional): Number of variational samples used for the Monte Carlo integraion. Defaults to 64.
            key_int (int, optional): Integer that is used as key for PRNG in JAX. Defaults to 1.
            chunk_size (int, optional): Chunk sizes of to evaluate. Defaults to 1.
            epoch (int, optional): Number of times that the learning algorithm will
            work thorugh the entire data. Defaults to 1000.

        Returns:
            Tuple: Last ELBO and the optimized variational parameters in a dictionary.
        """

        optimizer = optax.adam(learning_rate=step_size)
        opt_state = optimizer.init(self.init_variational_params)

        def bbvi_body(state, batch_idx, key):

            value, grads = jax.value_and_grad(self.lower_bound)(state.params, batch_idx, num_var_samples, key[0])
            updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
            params = optax.apply_updates(state.params, updates)

            return BbviState(key[1], opt_state, params), -value

        def epoch_body(state, scan_input):

            key, subkey = jax.random.split(state.key)
            rand_perm = jax.random.permutation(subkey, self.num_obs)

            for i in range(self.num_obs // batch_size):
                key, *subkey = jax.random.split(key, 3)

                if i != self.num_obs // batch_size:
                    batch_idx =  rand_perm[i * batch_size : (i + 1) * batch_size]
                    state, elbo = bbvi_body(state, batch_idx, subkey)
                elif i == self.num_obs // batch_size:
                    batch_idx =  rand_perm[i * batch_size : -1]
                    state, elbo = bbvi_body(state, batch_idx, subkey)

            return state, elbo

        @partial(jax.jit, static_argnums=0)
        def jscan(chunk_size: int, bbvi_state: BbviState) -> Tuple[BbviState, jnp.array]:

            scan_input = jnp.arange(chunk_size)
            new_state, elbo_chunk = jax.lax.scan(epoch_body, bbvi_state, scan_input)

            return new_state, elbo_chunk

        key = jax.random.PRNGKey(key_int)

        state = BbviState(key,
                          opt_state,
                          self.init_variational_params)

        elbo_chunks = []

        for _ in range(epochs // chunk_size):
            state, elbo_chunk = jscan(chunk_size, state)
            elbo_chunks.append(elbo_chunk)
            elbo_delta = abs(jnp.mean(elbo_chunk) - jnp.mean(elbo_chunks[-5])) if len(elbo_chunks) > 5 else jnp.inf
            if elbo_delta < threshold:
                break

        self.elbo_hist["elbo"] = jnp.concatenate(elbo_chunks)
        self.elbo_hist["epoch"] = jnp.arange(len(self.elbo_hist["elbo"]))
        self.variational_params = state.params
        self.set_opt_variational_params()

        return self.elbo_hist["elbo"][-1], self.opt_variational_params

    def set_opt_variational_params(self):
        """
        Method to obtain the variational parameters in terms of the covariance matrix and not the lower cholesky factor.
        """

        for node in self.graph.prob_traversal_order:
            node_type = self.digraph.nodes[node].get("node_type")
            if node_type == "strong":
                self.opt_variational_params[node] = {
                    "loc": self.variational_params[node]["loc"],
                    "cov": jnp.linalg.inv(jnp.dot(self.variational_params[node]["lower_tri"], self.variational_params[node]["lower_tri"].T))
                }

    def plot_elbo(self):
        """
        Method to visualize the progression of the ELBO during the optimization.
        """

        plt.plot(self.elbo_hist["epoch"], self.elbo_hist["elbo"])
        plt.title("Progression of the ELBO")
        plt.xlabel("Epoch")
        plt.ylabel("ELBO")
        plt.show()
