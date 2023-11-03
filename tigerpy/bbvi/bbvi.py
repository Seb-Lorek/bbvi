"""
Black-box variational inference.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

import tensorflow_probability.substrates.jax.distributions as tfjd
import tensorflow_probability.substrates.numpy.distributions as tfnd

import networkx as nx
import matplotlib.pyplot as plt
import functools
from functools import partial
import copy
from typing import (
    List,
    Dict,
    Tuple,
    NamedTuple,
    Callable,
    Union,
)

from .transform import (
    log_transform,
    exp_transform,
    batched_jac_determinant,
    log_cholesky_parametrization_to_tril,
    cov_from_prec_chol
)

from ..model.model import (
    Distribution,
    Array,
    Any
)

from ..model.nodes import (
    ModelGraph,
)

from ..distributions.mvn import (
    mvn_precision_chol_log_prob,
    mvn_precision_chol_sample,
    mvn_sample_noise
)

from ..distributions.mvn_log import (
    mvn_log_precision_chol_log_prob,
    mvn_log_precision_chol_sample,
    mvn_log_mean,
    mvn_log_cov
)

from .init_bbvi import (
    set_init_var_params,
    set_init_loc_scale,
    add_jitter
)

class BbviState(NamedTuple):
    data: dict
    batches: list
    num_obs: int
    key: jax.random.PRNGKey
    params: dict
    opt_state: Any
    params_new: dict
    elbo: jax.Array
    grad: dict

class EpochState(NamedTuple):
    data: dict
    elbo_best: jax.Array
    params_best: dict
    key: jax.random.PRNGKey
    params: dict
    opt_state: Any
    params_new: dict

class Bbvi:
    """
    Inference algorithm.
    """

    def __init__(self, 
                 graph: ModelGraph, 
                 jitter_init: bool=True,
                 model_init: bool=False,
                 loc_prec: float=1.0,
                 scale_prec: float=2.0) -> None:
        self.graph = copy.deepcopy(graph)
        self.digraph = self.graph.digraph
        self.model = self.graph.model
        self.num_obs = self.model.num_obs
        self.data = {}
        self.jitter_init = jitter_init
        self.model_init = model_init
        self.init_var_params = {}
        self.var_params = {}
        self.num_var_params = None
        self.trans_var_params = {}
        self.return_loc_params = {}
        self.elbo_hist = {}
        self.elbo_hist_burn = {}

        self.set_data()
        self.set_var_params(loc_prec, 
                            scale_prec)
        self.count_var_params()

    def set_data(self) -> None:
        """
        Method to gather the data (fixed node values and the response node value) from the DAG.
        """

        for node in self.graph.traversal_order:
            node_type = self.digraph.nodes[node]["node_type"]
            attr = self.digraph.nodes[node]["attr"]
            if node_type == "fixed":
                self.data[node] = attr["value"]
            elif node_type == "root":
                self.data[node] = attr["value"]

    def set_var_params(self, 
                       loc_prec: float=1.0,
                       scale_prec: float=2.0) -> None:
        """
        Method to set the internal variational parameters.
        """

        if self.model_init == True:
            if self.digraph.nodes["response"]["attr"]["dist"] is tfjd.Normal:
                self.init_var_params = set_init_loc_scale(self.digraph,
                                                          self.graph.prob_traversal_order,
                                                          loc_prec,
                                                          scale_prec)
        else:
            for node in self.graph.prob_traversal_order:
                node_type = self.digraph.nodes[node]["node_type"]
                if node_type == "strong":
                    attr = self.digraph.nodes[node]["attr"]
                    self.init_var_params[node] = set_init_var_params(attr,
                                                                     loc_prec,
                                                                     scale_prec)

    def count_var_params(self) -> None:
        """
        Method to count the number of variational parameters.
        """

        num_params = 0
        for item in self.init_var_params.values():
            num_params += item["loc"].shape[0]
            num_params += item["log_cholesky_prec"].shape[0]
        
        self.num_var_params = num_params

    @staticmethod
    def calc_lpred(design_matrix: jax.Array, 
                   params: Dict,
                   bijector: Callable) -> jax.Array:
        """
        Method to calculate the values for a linear predictor.

        Args:
            design_matrix (Dict): jax.Array that contains a design matrix to calculate the linear predictor.
            params (Dict): Parameters of the linear predictor in a dictionary using new variational samples.
            bijector (Callable): The inverse link function that transform the linear predictor 
            to the appropriate parameter space.

        Returns:
            jax.Array: The linear predictor at the new variational samples.
        """

        batch_design_matrix = jnp.expand_dims(design_matrix, axis=0)
        array_params = jnp.concatenate([param for param in params.values()], axis=-1)
        batch_params = jnp.expand_dims(array_params, -1)
        batch_nu = jnp.matmul(batch_design_matrix, batch_params)
        nu = jnp.squeeze(batch_nu)
 
        if bijector is not None:
            transformed = bijector(nu)
        else:
            transformed = nu

        return transformed

    @staticmethod
    def calc_scaled_loglik(log_lik: jax.Array,
                           num_obs: int) -> jax.Array:
        """
        Method to caluclate the scaled log-lik.

        Args:
            log_lik (jax.Array): Log-likelihood of the model.
            num_obs (int): Number of observations in the model

        Returns:
            scaled_log_lik: The scaled subsampled log-likelhood.
        """

        scaled_log_lik = num_obs * jnp.mean(log_lik, axis=-1)

        return scaled_log_lik
    
    @staticmethod
    def init_dist(dist: Distribution,
                  params: dict) -> Distribution:
        """
        Method to initialize the probability distribution of a strong node.

        Args:
            dist (Distribution): A tensorflow probability distribution.
            params (dict): Key, value pair, where keys should match the names of the parameters of
            the distribution.

        Returns:
            Distribution: A initialized tensorflow probability distribution.
        """

        initialized_dist = dist(**params)

        return initialized_dist

    @staticmethod
    def logprior(dist: Distribution, value: jax.Array) -> jax.Array:
        """
        Method to calculate the log-prior probability of a parameter node (strong node).

        Args:
            dist (Distribution): A initialized tensorflow probability distribution.
            value (jax.Array): Value of the parameter.

        Returns:
            jax.Array: Prior log-probabilities of the parameters.
        """

        return dist.log_prob(value)

    @staticmethod
    def loglik(dist: Distribution,
               value: jax.Array) -> jax.Array:
        """
        Method to calculate the log-likelihood of the response (root node).

        Args:
            dist (Distribution): A initialized tensorflow probability distribution.
            value (jax.Array): Values of the response.

        Returns:
            jax.Array: Log-likelihood of the response.
        """

        return dist.log_prob(value)

    def mc_logprob(self, 
                   batch_data: Dict,
                   samples: Dict,
                   num_obs: int) -> jax.Array:
        """
        Calculate the Monte Carlo integral for the joint log-probability of the model.

        Args:
            batch_data (Dict): A subsample of the data (mini-batch).
            samples (Dict): Samples from the variational distribution.
            num_obs (int): Number of observations in the model.

        Returns:
            jax.Array: Monte carlo integral for the noisy log-probability of the model. We only use a subsample
            of the data.
        """

        # Safe intermediary computations in the model_state
        model_state = {}

        # Safe all log priors 
        # Maybe explicity define shape log-priors via num_var_samples
        # However gets recast correctly 
        log_priors = jnp.array([0.0])

        # Acess global variables but don't modify them 
        for node in self.graph.update_traversal_order:
            node_type = self.digraph.nodes[node]["node_type"]
            attr = self.digraph.nodes[node]["attr"]
            childs = list(self.digraph.successors(node))
            parents = list(self.digraph.predecessors(node))
            # Linear predictor node 
            if node_type == "lpred":
                for parent in parents:
                    edge = self.digraph.get_edge_data(parent, node)
                    if edge["role"] == "fixed":
                        design_matrix = batch_data[parent]
                        parents.remove(parent)

                # Obtain all parameters of the linear predictor node must be defined in the 
                # right order 
                params = {kw: samples[kw] for kw in parents}
               
                # Calculate the linear predictor with the new samples
                lpred_val = Bbvi.calc_lpred(design_matrix, 
                                            params,
                                            attr["bijector"])
                model_state[node] = lpred_val   
            # Strong node          
            elif node_type == "strong":
                if self.digraph.nodes[node]["input_fixed"]:
                    log_prior = Bbvi.logprior(attr["dist"], 
                                              samples[node])
                else: 
                    # Combine fixed hyperparameter with parameter that is stochastic
                    params = {}
                    for parent in parents:
                        if self.digraph.nodes[parent]["node_type"] == "hyper":
                            edge = self.digraph.get_edge_data(parent, node)
                            params[edge["role"]] = self.digraph.nodes[parent]["attr"]["value"]
                        elif self.digraph.nodes[parent]["node_type"] == "strong":
                            edge = self.digraph.get_edge_data(parent, node)
                            params[edge["role"]] = model_state[parent]
                    init_dist = Bbvi.init_dist(dist=attr["dist"], 
                                               params=params)
                    log_prior = Bbvi.logprior(init_dist, 
                                              samples[node])
                # Sum log-priors of a strong node
                if log_prior.ndim == 2:
                    log_prior = jnp.sum(log_prior, axis=-1)
                log_priors += log_prior
                model_state[node] = samples[node]
            # Root node 
            elif node_type == "root":
                params = {}
                for parent in parents:
                        edge = self.digraph.get_edge_data(parent, node)
                        params[edge["role"]] = model_state[parent]
                # Comments here refer to instability of location-scale-shape regression
                # replace the scale coefficients just with a one 
                # print("Scale-mean", params["scale"], params["scale"].shape)
                # print("Scale-mean over batches", jnp.mean(params["scale"], axis=0), jnp.mean(params["scale"], axis=0).shape)
                
                # Write function that initializes the dist with new values
                init_dist = Bbvi.init_dist(dist=attr["dist"],
                                           params=params)
                # print("Response dist:", init_dist)

                # Calculate the log_lik
                log_lik = Bbvi.loglik(init_dist, 
                                      batch_data[node])
                # print("Log-lik:", log_lik, log_lik.shape)
                # print("Mean log-lik", jnp.mean(log_lik, axis=-1), jnp.mean(log_lik, axis=-1).shape)
                # print("Scaled log-lik", num_obs * jnp.mean(log_lik, axis=-1), jnp.mean(log_lik, axis=-1).shape)
                # calculate the scaled log-likelihood
                scaled_log_lik = Bbvi.calc_scaled_loglik(log_lik,
                                                         num_obs)
        
        # print("Scaled log-lik", jnp.mean(scaled_log_lik), jnp.mean(scaled_log_lik).shape)
        # print("Log-priors", jnp.mean(log_priors), jnp.mean(log_priors).shape)
        
        return jnp.mean(scaled_log_lik + log_priors)

    @staticmethod 
    def neg_entropy_unconstr(var_params: Dict,
                             samples_params: Dict,
                             samples_noise: Dict,
                             var: str,
                             num_var_samples: int,
                             key: jax.random.PRNGKey) -> Tuple[Dict, Array]:
        
        loc, log_cholesky_prec = var_params[var]["loc"], var_params[var]["log_cholesky_prec"]
        lower_tri = log_cholesky_parametrization_to_tril(log_cholesky_prec, d=loc.shape[0])
        noise = mvn_sample_noise(key=key, 
                                 shape=lower_tri.shape,
                                 S=num_var_samples)
        s = mvn_precision_chol_sample(loc=loc, 
                                      precision_matrix_chol=lower_tri, 
                                      noise=noise)
        l = mvn_precision_chol_log_prob(x=s, 
                                        loc=loc, 
                                        precision_matrix_chol=lower_tri)
        samples_params[var] = s
        samples_noise[var] = noise
        neg_entropy = jnp.mean(l, keepdims=True)

        return samples_params, samples_noise, neg_entropy

    @staticmethod 
    def neg_entropy_posconstr(var_params: Dict, 
                               samples_params: Dict, 
                               samples_noise: Dict, 
                               var: str, 
                               num_var_samples: int,
                               key: jax.random.PRNGKey) -> Tuple[Dict, Array]:
        
        loc, log_cholesky_prec = var_params[var]["loc"], var_params[var]["log_cholesky_prec"]
        lower_tri = log_cholesky_parametrization_to_tril(log_cholesky_prec, d=loc.shape[0])
        noise = mvn_sample_noise(key=key,
                                 shape=lower_tri.shape,
                                 S=num_var_samples)
        s = mvn_log_precision_chol_sample(loc=loc, 
                                          precision_matrix_chol=lower_tri, 
                                          noise=noise)
        l = mvn_log_precision_chol_log_prob(x=s, 
                                            loc=loc, 
                                            precision_matrix_chol=lower_tri)
        samples_params[var] = s
        samples_noise[var] = noise
        neg_entropy = jnp.mean(l, keepdims=True)

        return samples_params, samples_noise, neg_entropy

    # Update docstrings
    def lower_bound(self, 
                    var_params: Dict,
                    batch_data: Dict,
                    num_obs: int,
                    num_var_samples: int,
                    key: jax.random.PRNGKey) -> Array:

        key, *subkeys = jax.random.split(key, len(var_params)+1)
        samples_params = {}
        samples_noise = {}
        total_neg_entropy = jnp.array([])
        i = 0
        for kw in var_params.keys():
            if self.digraph.nodes[kw]["attr"]["param_space"] is None:
                samples_params, samples_noise, neg_entropy = Bbvi.neg_entropy_unconstr(var_params, 
                                                                                       samples_params,
                                                                                       samples_noise, 
                                                                                       kw, 
                                                                                       num_var_samples,
                                                                                       subkeys[i])
                total_neg_entropy = jnp.append(total_neg_entropy, neg_entropy, axis=0)
            elif self.digraph.nodes[kw]["attr"]["param_space"] == "positive":
                samples_params, samples_noise, neg_entropy = Bbvi.neg_entropy_posconstr(var_params, 
                                                                                        samples_params,
                                                                                        samples_noise, 
                                                                                        kw,
                                                                                        num_var_samples, 
                                                                                        subkeys[i])
                total_neg_entropy = jnp.append(total_neg_entropy, neg_entropy, axis=0)
            i += 1
        # print(samples["tau2"], samples["eta2"])
        # print(total_neg_entropy)
        mc_log_prob = self.mc_logprob(batch_data, samples_params, num_obs)
        elbo = mc_log_prob - jnp.sum(total_neg_entropy)
        
        return - elbo, samples_noise

    # experimental
    def calc_control_variate(self, 
                             var_params: Dict,
                             batch_data: Dict,
                             noise_samples: Dict,
                             num_obs: int,
                             key: jax.random.PRNGKey):

        loc_params = {kw: value["loc"] for kw, value in var_params.items()}
        log_chol_prec_params = {kw: value["log_cholesky_prec"] for kw, value in var_params.items()}
        first = jax.grad(self.mc_logprob, argnums=1)(batch_data,
                                                           loc_params,
                                                           num_obs)
        second = hessian(self.mc_logprob, argnums=1)(batch_data,
                                                           loc_params,
                                                           num_obs)
        
        diff = mvn_precision_chol_sample(loc, 
                                        precision_chol,
                                        noise)

        jac_map = jax.jacfwd(mvn.mvn_precision_chol_sample, argnums=(0,1))

        mvn_precision_chol_sample
        combined_second_order = second_order
        grad_control = jax.tree_map(lambda x,y: x + y, first_order, combined_second_order)
        grad_control = jax.tree_map(lambda x: jnp.mean(x), grad_control)
        grad_control_mean = data_term + 0
        
        return jax.tree_map(lambda x,y: x - y, grad_control, grad_control_mean)

    def run_bbvi(self,
                 step_size: Union[Any, float]=1e-3,
                 threshold: float=1e-2,
                 key_int: int=1,
                 batch_size: int=64,
                 num_var_samples=32,
                 chunk_size: int=1,
                 epochs: int=500) -> tuple:
        """
        Method to start the stochastic gradient optimization. The implementation uses Adam.

        Args:
            step_size (Union[Any, float], optional): Step size (learning rate) of the SGD optimizer (Adam).
            Can also be a scheduler. Defaults to 1e-3.
            threshold (float, optional): Threshold to stop the optimization. Defaults to 1e-2.
            key_int (int, optional): Integer that is used as argument for PRNG in JAX. Defaults to 1.
            batch_size (int, optional): Batchsize, i.e. number of samples to use during SGD. Defaults to 64.
            num_var_samples (int, optional): Number of variational samples used for the Monte Carlo integration. Defaults to 64.
            chunk_size (int, optional): Chunk sizes over which jax.lax.scan scans, 1 results in a classic for loop. Defaults to 1.
            epoch (int, optional): Number of iterations loops of the optimization algorithm such that 
            the algorithm has iterated each time through the entire dataset. Defaults to 500.

        Returns:
            Tuple: Last ELBO (jnp.float32) and the optimized variational parameters (dict).
        """

        if type(step_size) is float:
            optimizer = optax.adam(learning_rate=step_size)
        else:
            optimizer = optax.adamw(learning_rate=step_size)
        
        def bbvi_body(idx, bbvi_state):
            
            key, subkey = jax.random.split(bbvi_state.key)
            
            funcs = [lambda i=i: bbvi_state.batches[i] for i in range(len(bbvi_state.batches))]
            batch_idx = jax.lax.switch(idx, funcs)
            batch_data = jax.tree_map(lambda x: x[batch_idx], bbvi_state.data)

            # Note: num_var_samples is a global variable
            lower_bound_grad_value = jax.value_and_grad(self.lower_bound, has_aux=True)
            aux, grad = lower_bound_grad_value(bbvi_state.params_new,
                                               batch_data,
                                               bbvi_state.num_obs,
                                               num_var_samples,
                                               subkey)
            neg_elbo, noise_samples = aux
            #cv = self.calc_control_variate(bbvi_state.params_new,
            #                               batch_data,
            #                               noise_samples,
            #                               bbvi_state.num_obs)
            #grad_cv = jax.tree_map(lambda x,y: x - y, grad, cv) 
            #replace with grad
            updates, opt_state = optimizer.update(grad, 
                                                  bbvi_state.opt_state, 
                                                  bbvi_state.params_new)
            params_new = optax.apply_updates(bbvi_state.params_new, 
                                             updates)
            elbo = - neg_elbo

            return BbviState(bbvi_state.data,
                             bbvi_state.batches,
                             bbvi_state.num_obs,
                             key,
                             bbvi_state.params_new,
                             opt_state,
                             params_new,
                             elbo,
                             grad)
                
        def epoch_body(epoch_state, scan_input):
            key, subkey = jax.random.split(epoch_state.key)
            num_obs = epoch_state.data["response"].shape[0]
            rand_perm = jax.random.permutation(subkey, num_obs)         
            # Note: batch_size is a global variable
            num_batches = num_obs // batch_size 
            lost_obs = num_obs % batch_size
            if lost_obs > 0:
                cut_rand_perm = jax.lax.slice_in_dim(rand_perm, start_index=0, limit_index=-lost_obs)
            else:
                cut_rand_perm = rand_perm
            batches = jnp.split(cut_rand_perm, num_batches)
            
            bbvi_state = BbviState(epoch_state.data,
                                   batches,
                                   num_obs,
                                   key,
                                   epoch_state.params,
                                   epoch_state.opt_state,
                                   epoch_state.params_new,
                                   jnp.array(0.0),
                                   epoch_state.params)

            # Note: batch_size is a global variable 
            bbvi_state = jax.lax.fori_loop(0, 
                                           num_obs // batch_size, 
                                           bbvi_body, 
                                           bbvi_state)
            
            # Choose max ELBO here and pass to EpochState 
            def true_fn(params, elbo, params_best, elbo_best):
                return params, elbo
            
            def false_fn(params, elbo, params_best, elbo_best):
                return params_best, elbo_best
            
            params_best, elbo_best = jax.lax.cond(bbvi_state.elbo > epoch_state.elbo_best,
                                                  true_fn,
                                                  false_fn,
                                                  bbvi_state.params,
                                                  bbvi_state.elbo,
                                                  epoch_state.params_best, 
                                                  epoch_state.elbo_best)
            
            return EpochState(epoch_state.data,
                              elbo_best,
                              params_best,
                              bbvi_state.key,
                              bbvi_state.params,
                              bbvi_state.opt_state,
                              bbvi_state.params_new), (bbvi_state.elbo, bbvi_state.grad)

        @partial(jax.jit, static_argnums=0)
        def jscan(chunk_size: int, epoch_state: EpochState) -> Tuple[EpochState, jnp.array]:

            scan_input = (jnp.arange(chunk_size), 
                          jax.tree_map(lambda x: jnp.broadcast_to(x, shape=(chunk_size, x.shape[0])), epoch_state.params))
            new_state, chunk = jax.lax.scan(epoch_body, epoch_state, scan_input)

            return new_state, chunk

        key = jax.random.PRNGKey(key_int)

        if self.jitter_init:
            key, subkey = jax.random.split(key)
            self.init_var_params = add_jitter(self.init_var_params, 
                                              subkey)

        opt_state = optimizer.init(self.init_var_params)

        epoch_state = EpochState(self.data,
                                 jnp.array(-jnp.inf),
                                 self.init_var_params,
                                 key,
                                 self.init_var_params,
                                 opt_state,
                                 self.init_var_params)
        
        elbo_history = jnp.array([])
        grad_history = []

        print("Start optimization ...")

        for _ in range(epochs // chunk_size):
            epoch_state, chunk = jscan(chunk_size, epoch_state)
            elbo_history = jnp.append(elbo_history, chunk[0], axis=0)
            grad_history.append(chunk[1])
            elbo_delta = abs(elbo_history[-1] - elbo_history[-200]) if len(elbo_history) > 200 else jnp.inf
            if elbo_delta < threshold:
                break

        print("Finished optimization.")

        self.elbo_hist["elbo"] = elbo_history
        self.elbo_hist["grad"] = grad_history
        self.elbo_hist["epoch"] = jnp.arange(elbo_history.shape[0])
        self.var_params = epoch_state.params_best
        self.set_trans_var_params()

    def set_trans_var_params(self):
        """
        Method to obtain the variational parameters in terms of the covariance matrix and not the lower cholesky factor.
        """

        for node in self.graph.prob_traversal_order:
            node_type = self.digraph.nodes[node]["node_type"]
            attr = self.digraph.nodes[node]["attr"]
            if node_type == "strong":
                loc = self.var_params[node]["loc"]
                lower_tri = log_cholesky_parametrization_to_tril(self.var_params[node]["log_cholesky_prec"], 
                                                                 d=loc.shape[0])
                cov =  cov_from_prec_chol(lower_tri)
                if attr["param_space"] == "positive":
                    self.trans_var_params[node] = {
                        "loc": mvn_log_mean(loc, cov),
                        "cov": mvn_log_cov(loc, cov)
                    }
                    self.return_loc_params[node] = {
                        "loc": mvn_log_mean(loc, cov)
                    }
                else:
                    self.trans_var_params[node] = {
                        "loc": loc,
                        "cov": cov
                    }
                    self.return_loc_params[node] = {
                        "loc": loc
                    }

    def plot_elbo(self):
        """
        Method to visualize the progression of the ELBO during the optimization.
        """

        plt.plot(self.elbo_hist["epoch"], self.elbo_hist["elbo"])
        plt.title("Progression of the ELBO during optimization")
        plt.xlabel("Epoch")
        plt.ylabel("ELBO")
        plt.show() 