"""
Black-box variational inference.
"""

import jax
import jax.numpy as jnp

import optax

import tensorflow_probability.substrates.jax.distributions as tfjd
import tensorflow_probability.substrates.numpy.distributions as tfnd

import matplotlib.pyplot as plt

import networkx as nx

import functools
from functools import partial

import copy

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
    Distribution,
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
    hyperparams: Dict
    num_obs: int
    key: jax.random.PRNGKey
    opt_state: Any
    params: Any

class EpochState(NamedTuple):
    data: Dict
    hyperparams: Dict
    key: jax.random.PRNGKey
    opt_state: Any
    params: Any


class Bbvi_bayes_lr:
    """Inference algorithm."""

    def __init__(self, graph: ModelGraph) -> None:
        self.graph = copy.deepcopy(graph)
        self.digraph = self.graph.digraph
        self.model = self.graph.model
        self.num_obs = self.model.num_obs

        self.data = {}
        self.hyperparams = {}
        self.param_space = {}

        self.init_variational_params = {}
        self.variational_params = {}
        self.opt_variational_params = {}
        self.elbo_hist = {}

        self.set_data()
        self.set_hyperparams()
        self.set_param_space()
        self.set_variational_params()

    def set_data(self) -> None:
        """
        Method to gather the data from the tree.
        """

        for node in self.graph.traversal_order:
            node_type = self.digraph.nodes[node]["node_type"]
            attr = self.digraph.nodes[node]["attr"]
            if node_type == "fixed":
                self.data[node] = attr["value"]
            elif node_type == "root":
                self.data[node] = attr["value"]

    def set_hyperparams(self) -> None:
        """
        Method to gather the hyperparameters from the tree.
        """

        initialized_successors = {}

        for node in self.graph.traversal_order:
            node_type = self.digraph.nodes[node]["node_type"]
            attr = self.digraph.nodes[node]["attr"]
            successors = list(self.digraph.successors(node))

            if successors:
                successor = successors[0]

            if node_type == "hyper":
                if successor not in initialized_successors:
                    self.hyperparams[successor] = {}
                    initialized_successors[successor] = True
                self.hyperparams[successor][node] = attr["value"]

    def set_param_space(self) -> None:
        """
        Method to set the parameter space.
        """

        for node in self.graph.traversal_order:
            node_type = self.digraph.nodes[node]["node_type"]
            attr = self.digraph.nodes[node]["attr"]
            if node_type == "strong":
                self.param_space[node] = attr["param_space"]

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

        if attr["param_space"] is None:
            loc = attr["value"]
            lower_tri = jnp.diag(jnp.ones(attr["dim"]))
        elif attr["param_space"] == "positive":
            loc = jnp.log(attr["value"])
            lower_tri = jnp.diag(jnp.ones(attr["dim"]))

        return {"loc": loc, "lower_tri": lower_tri}

    @staticmethod
    def calc_lpred(data: Dict, samples: Dict) -> Array:
        """
        Method to calculate the linear predictor.

        Args:
            data (Dict): Dictionary that contains the data.
            samples (Dict): Samples of the variational distribution

        Returns:
            Array: the linear predictor at the new variational samples.
        """

        design_matrix = data["X"]
        design_matrix = jnp.expand_dims(design_matrix, axis=0)

        values_params = samples["beta"]
        values_params = jnp.expand_dims(values_params, -1)
        nu = jnp.matmul(design_matrix, values_params)
        nu = jnp.squeeze(nu)

        return nu

    @staticmethod
    def calc_scaled_loglik(log_lik: Array,
                           num_obs: int) -> Array:
        """
        Caluclate the scaled log-lik.

        Args:
            log_lik (Array): Log-likelihood of the model.
            num_obs (int): Number of observations in the model

        Returns:
            scaled_log_lik: The scaled subsampled log-likelhood.
        """

        scaled_log_lik = num_obs * jnp.mean(log_lik, axis=-1)

        return scaled_log_lik

    @staticmethod
    def logprior(dist: Distribution, value: Array) -> Array:
        """
        Method to calculate the log-prior probability of a parameter node (strong node).

        Args:
            dist (Distribution): A initialized tensorflow probability distribution.
            value (Array): Value of the parameter.

        Returns:
            Array: Prior log-probabilities of the parameters.
        """

        return dist.log_prob(value)

    @staticmethod
    def loglik(dist: Distribution,
               value: Array) -> Array:
        """
        Method to calculate the log-likelihood of the response (root node).

        Args:
            dist (Distribution): A initialized tensorflow probability distribution.
            value (Array): Values of the response.

        Returns:
            Array: Log-likelihood of the response.
        """

        return dist.log_prob(value)

    @staticmethod
    def mc_logprob(data: Dict,
                   hyperparams: Dict,
                   samples: Dict,
                   num_obs: int) -> Array:
        """
        Calculate the Monte Carlo integral for the joint log-probability of the model.

        Args:
            samples (Dict): Samples from the variational distribution for the parameters of the model in a dictionary.
            batch_idx (Array): Indexes of the mini-batch.
            num_var_samples(int): Number of samples from the variational distribution.

        Returns:
            Array: Monte carlo integral for the noisy log-probability of the model. We only use a subsample
            of the data.
        """

        # calculate the linear predictor
        lpred = Bbvi_bayes_lr.calc_lpred(data, samples)

        # response dist
        dist_response = tfjd.Bernoulli(logits=lpred)

        # calculate the log-likelihood
        log_lik = Bbvi_bayes_lr.loglik(dist_response, data["response"])

        # calculate the scaled log-likelihood
        scaled_log_lik = Bbvi_bayes_lr.calc_scaled_loglik(log_lik,
                                                          num_obs)
        # print("Scaled log-lik", scaled_log_lik.shape)

        beta_loc = hyperparams["beta"]["beta_loc"]
        beta_scale = hyperparams["beta"]["beta_scale"]
        dist_beta = tfjd.Normal(loc=beta_loc, scale=beta_scale)

        log_prior_beta = Bbvi_bayes_lr.logprior(dist_beta, samples["beta"])

        # print("Log-prior-beta:", log_prior_beta.shape)

        log_prior = jnp.sum(log_prior_beta, axis=-1)
        # print("Log-prior:", log_prior.shape)

        return jnp.mean(scaled_log_lik + log_prior)

    @staticmethod
    def lower_bound(variational_params: Dict,
                    data: Dict,
                    hyperparams: Dict,
                    num_var_samples: int,
                    num_obs: int,
                    key: jax.random.PRNGKey) -> Array:
        """
        Method to calculate the negative ELBO (evidence lower bound).

        Args:
            variational_params (Dict): The varational paramters in a nested dictionary where the first key identifies the paramter of the model.
            batch_idx (Array): Indexes of the mini-batch.
            num_var_samples(int): Number of samples from the variational distribution.
            key (jax.random.PRNGKey): A pseudo-random number generation key from JAX.

        Returns:
            Array: Negative ELBO.
        """

        key, *subkeys = jax.random.split(key, len(variational_params)+1)

        samples = {}

        loc, lower_tri = variational_params["beta"]["loc"], variational_params["beta"]["lower_tri"]

        lower_tri = jnp.tril(lower_tri)

        samples["beta"] = mvn_precision_chol_sample(loc=loc, precision_matrix_chol=lower_tri, key=subkeys[0], S=num_var_samples)
        l = mvn_precision_chol_log_prob(x=samples["beta"], loc=loc, precision_matrix_chol=lower_tri)

        neg_entropy = jnp.mean(l)

        mc_log_prob = Bbvi_bayes_lr.mc_logprob(data, hyperparams, samples, num_obs)

        elbo = mc_log_prob - neg_entropy

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

        def bbvi_body(bbvi_state, data, key):

            neg_elbo, grads = jax.value_and_grad(Bbvi_bayes_lr.lower_bound)(bbvi_state.params,
                                                                            data,
                                                                            bbvi_state.hyperparams,
                                                                            num_var_samples,
                                                                            bbvi_state.num_obs,
                                                                            key)
            updates, opt_state = optimizer.update(grads, bbvi_state.opt_state, bbvi_state.params)
            params = optax.apply_updates(bbvi_state.params, updates)

            return BbviState(bbvi_state.hyperparams,
                             bbvi_state.num_obs,
                             key,
                             opt_state,
                             params), - neg_elbo

        def epoch_body(epoch_state, scan_input):

            key, subkey = jax.random.split(epoch_state.key)
            num_obs = epoch_state.data["response"].shape[0]
            rand_perm = jax.random.permutation(subkey, num_obs)

            bbvi_state = BbviState(epoch_state.hyperparams,
                                   num_obs,
                                   key,
                                   epoch_state.opt_state,
                                   epoch_state.params)

            for i in range(num_obs // batch_size):
                key, subkey = jax.random.split(key)

                if i != num_obs // batch_size:
                    batch_idx =  rand_perm[i * batch_size : (i + 1) * batch_size]
                    data = jax.tree_map(lambda x: x[batch_idx], epoch_state.data)
                    bbvi_state, elbo = bbvi_body(bbvi_state, data, subkey)
                elif i == num_obs // batch_size:
                    batch_idx =  rand_perm[i * batch_size : -1]
                    data = jax.tree_map(lambda x: x[batch_idx], epoch_state.data)
                    bbvi_state, elbo = bbvi_body(bbvi_state, data, subkey)

            return EpochState(epoch_state.data,
                              epoch_state.hyperparams,
                              key,
                              bbvi_state.opt_state,
                              bbvi_state.params), elbo

        @partial(jax.jit, static_argnums=0)
        def jscan(chunk_size: int, epoch_state: EpochState) -> Tuple[EpochState, jnp.array]:

            scan_input = jnp.arange(chunk_size)
            new_state, elbo_chunk = jax.lax.scan(epoch_body, epoch_state, scan_input)

            return new_state, elbo_chunk

        key = jax.random.PRNGKey(key_int)

        epoch_state = EpochState(self.data,
                                 self.hyperparams,
                                 key,
                                 opt_state,
                                 self.init_variational_params)

        elbo_history = jnp.array([])

        for _ in range(epochs // chunk_size):
            epoch_state, elbo_chunk = jscan(chunk_size, epoch_state)
            elbo_history = jnp.append(elbo_history, elbo_chunk, axis=0)
            elbo_delta = abs(elbo_history[-1] - elbo_history[-200]) if len(elbo_history) > 200 else jnp.inf
            if elbo_delta < threshold:
                break

        self.elbo_hist["elbo"] = elbo_history
        self.elbo_hist["epoch"] = jnp.arange(elbo_history.shape[0])
        self.variational_params = epoch_state.params
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
