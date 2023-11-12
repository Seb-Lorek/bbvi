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
import time 
from typing import (
    List,
    Dict,
    Tuple,
    NamedTuple,
    Callable,
    Union,
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

from .transform import (
    log_transform,
    exp_transform,
    batched_jac_determinant,
    log_cholesky_parametrization_to_tril,
    cov_from_prec_chol
)

from .init_bbvi import (
    set_init_var_params,
    set_init_loc_scale,
    add_jitter
)

class BbviState(NamedTuple):
    data_train: dict
    data_val: dict
    batches: list
    num_obs_train: int
    num_obs_val: int 
    key: jax.random.PRNGKey
    opt_state: Any
    params: dict
    elbo_train: jax.Array
    elbo_val: jax.Array
    grad: dict
    it: int

class EpochState(NamedTuple):
    data_train: dict
    data_val: dict
    num_obs_train: int
    num_obs_val: int 
    elbo_best: jax.Array
    params_best: dict
    key: jax.random.PRNGKey
    opt_state: Any
    params: dict
    elbo_train: jax.Array
    elbo_val: jax.Array
    it: int

class Bbvi:
    """
    Inference algorithm.
    """

    def __init__(self, 
                 graph: ModelGraph, 
                 jitter_init: bool=True,
                 model_init: bool=False,
                 loc_prec: float=1.0,
                 scale_prec: float=2.0, 
                 verbose=True) -> None:
        self.graph = copy.deepcopy(graph)
        self.digraph = self.graph.digraph
        self.model = self.graph.model
        self.num_obs = self.model.num_obs
        self.data = {}
        self.jitter_init = jitter_init
        self.model_init = model_init
        self.verbose = verbose
        self.init_var_params = {}
        self.var_params = {}
        self.num_var_params = None
        self.trans_var_params = {}
        self.return_loc_params = {}
        self.elbo_hist = {}

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
                   data: Dict,
                   samples: Dict,
                   num_obs: int) -> jax.Array:
        """
        Calculate the Monte Carlo integral for the joint log-probability of the model.

        Args:
            data (Dict): A subsample of the data.
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
                        design_matrix = data[parent]
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
                                      data[node])
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
                             var: str) -> Tuple[Dict, Array]:
        
        loc, log_cholesky_prec = var_params[var]["loc"], var_params[var]["log_cholesky_prec"]
        lower_tri = log_cholesky_parametrization_to_tril(log_cholesky_prec, d=loc.shape[0])
        s = mvn_precision_chol_sample(loc=loc, 
                                      precision_matrix_chol=lower_tri, 
                                      noise=samples_noise[var])
        l = mvn_precision_chol_log_prob(x=s, 
                                        loc=loc, 
                                        precision_matrix_chol=lower_tri)
        samples_params[var] = s
        neg_entropy = jnp.mean(jnp.atleast_1d(l), keepdims=True)

        return samples_params, neg_entropy

    @staticmethod 
    def neg_entropy_posconstr(var_params: Dict, 
                               samples_params: Dict, 
                               samples_noise: Dict, 
                               var: str) -> Tuple[Dict, Array]:
        
        loc, log_cholesky_prec = var_params[var]["loc"], var_params[var]["log_cholesky_prec"]
        lower_tri = log_cholesky_parametrization_to_tril(log_cholesky_prec, d=loc.shape[0])
        s = mvn_log_precision_chol_sample(loc=loc, 
                                          precision_matrix_chol=lower_tri, 
                                          noise=samples_noise[var])
        l = mvn_log_precision_chol_log_prob(x=s, 
                                            loc=loc, 
                                            precision_matrix_chol=lower_tri)
        samples_params[var] = s

        neg_entropy = jnp.mean(jnp.atleast_1d(l), keepdims=True)

        return samples_params, neg_entropy

    # Update docstrings
    def lower_bound(self, 
                    var_params: Dict,
                    data: Dict,
                    num_obs: int,
                    samples_noise: Dict) -> Array:

        samples_params = {}
        total_neg_entropy = jnp.array([])
        i = 0
        for kw in var_params.keys():
            if self.digraph.nodes[kw]["attr"]["param_space"] is None:
                samples_params, neg_entropy = Bbvi.neg_entropy_unconstr(var_params, 
                                                                        samples_params,
                                                                        samples_noise, 
                                                                        kw)                                                                           
                total_neg_entropy = jnp.append(total_neg_entropy, neg_entropy, axis=0)
            elif self.digraph.nodes[kw]["attr"]["param_space"] == "positive":
                samples_params, neg_entropy = Bbvi.neg_entropy_posconstr(var_params, 
                                                                         samples_params,
                                                                         samples_noise, 
                                                                         kw)
                total_neg_entropy = jnp.append(total_neg_entropy, neg_entropy, axis=0)
            i += 1

        # print(samples["tau2"], samples["eta2"])
        # print(total_neg_entropy)

        mc_log_prob = self.mc_logprob(data, 
                                      samples_params, 
                                      num_obs)
        elbo = mc_log_prob - jnp.sum(total_neg_entropy)
        
        return - elbo

    @staticmethod
    def gen_noise(var_params, 
                  num_var_samples, 
                  key) -> Dict:

        samples_noise = {}
        key, *subkeys = jax.random.split(key, len(var_params)+1)

        i = 0
        for kw in var_params.keys():
            samples_noise[kw] = mvn_sample_noise(key=subkeys[i],
                                                 shape= var_params[kw]["loc"].shape,
                                                 S=num_var_samples)
            i += 1

        return samples_noise
    
    # Update Docstrings 
    def run_bbvi(self,
                 step_size: Union[Any, float]=1e-3,
                 grad_clip: float=1,
                 threshold: float=1e-2,
                 key_int: int=42,
                 batch_size: int=64,
                 train_share: float=0.8,
                 num_var_samples=32,
                 chunk_size: int=1,
                 epochs: int=500) -> tuple:
        """
        Method to start the stochastic gradient optimization. The implementation uses Adam.

        Args:
            step_size (Union[Any, float], optional): Step size (learning rate) of the SGD optimizer (Adam).
            Can also be a scheduler. Defaults to 1e-3.
            grad_clip (float, optional): Float to determine the gradient clipping value.
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
            optimizer = optax.chain(optax.clip(grad_clip), 
                                    optax.adam(learning_rate=step_size))
        else:
            optimizer = optax.clip(optax.clip(grad_clip), 
                                   optax.adamw(learning_rate=step_size))
        
        def bbvi_body(idx, bbvi_state):
            
            key, subkey = jax.random.split(bbvi_state.key)
            funcs = [lambda i=i: bbvi_state.batches[i] for i in range(len(bbvi_state.batches))]
            batch_idx = jax.lax.switch(idx, funcs)
            batch_data = jax.tree_map(lambda x: x[batch_idx], bbvi_state.data_train)
            
            # Generate noise samples 
            samples_noise = Bbvi.gen_noise(bbvi_state.params,    
                                           num_var_samples,   
                                           subkey)

            # Note: num_var_samples is a global variable
            # Grad of the (neg) ELBO
            lower_bound_grad_value = jax.value_and_grad(self.lower_bound)
            neg_elbo_train, grad = lower_bound_grad_value(bbvi_state.params,
                                                          batch_data,
                                                          bbvi_state.num_obs_train,
                                                          samples_noise)

            # Update the variational parameters
            updates, opt_state = optimizer.update(grad, 
                                                  bbvi_state.opt_state, 
                                                  bbvi_state.params)
            params_new = optax.apply_updates(bbvi_state.params, 
                                             updates)
            
            # Calculate the ELBO for the validation data
            neg_elbo_val = self.lower_bound(params_new,
                                            bbvi_state.data_val,
                                            bbvi_state.num_obs_val,
                                            samples_noise)

            # Store the ELBO                    
            elbo_train = bbvi_state.elbo_train.at[bbvi_state.it].set(-neg_elbo_train)
            elbo_val = bbvi_state.elbo_val.at[bbvi_state.it].set(-neg_elbo_val)

            return BbviState(bbvi_state.data_train,
                             bbvi_state.data_val,
                             bbvi_state.batches,
                             bbvi_state.num_obs_train,
                             bbvi_state.num_obs_val,
                             key,
                             opt_state,
                             params_new,
                             elbo_train,
                             elbo_val,
                             grad, 
                             bbvi_state.it + 1)
                
        def epoch_body(epoch_state, scan_input):
            key, *subkeys = jax.random.split(epoch_state.key, 3)

            # Split data into train batches 
            num_obs_train = epoch_state.data_train["response"].shape[0]
            train_idx = jax.random.permutation(subkeys[0], num_obs_train)

            # Note: batch_size is a global variable
            num_batches = num_obs_train // batch_size 
            lost_obs = num_obs_train % batch_size
            if lost_obs > 0:
                add_idx = jax.random.choice(subkeys[1], train_idx[:-lost_obs], (batch_size-lost_obs,), replace=False)
                train_idx = jnp.append(train_idx, add_idx) 
                batches = jnp.split(train_idx, (num_batches+1))
            else:
                batches = jnp.split(train_idx, num_batches)

            bbvi_state = BbviState(epoch_state.data_train,
                                   epoch_state.data_val,
                                   batches,
                                   epoch_state.num_obs_train,
                                   epoch_state.num_obs_val,
                                   key,
                                   epoch_state.opt_state,
                                   epoch_state.params,
                                   epoch_state.elbo_train,
                                   epoch_state.elbo_val,
                                   epoch_state.params, 
                                   epoch_state.it)

            # Note: batch_size is a global variable 
            bbvi_state = jax.lax.fori_loop(0, 
                                           len(batches), 
                                           bbvi_body, 
                                           bbvi_state)
            
            # The current elbo and its param configuration
            curr_elbo = bbvi_state.elbo_val[bbvi_state.it - 1]

            # Choose max ELBO here and pass to EpochState 
            def true_fn(params, elbo, params_best, elbo_best):
                return params, elbo
            
            def false_fn(params, elbo, params_best, elbo_best):
                return params_best, elbo_best
            
            params_best, elbo_best = jax.lax.cond(curr_elbo > epoch_state.elbo_best,
                                                  true_fn,
                                                  false_fn,
                                                  bbvi_state.params,
                                                  curr_elbo,
                                                  epoch_state.params_best, 
                                                  epoch_state.elbo_best)

            return EpochState(epoch_state.data_train,
                              epoch_state.data_val,
                              epoch_state.num_obs_train,
                              epoch_state.num_obs_val,
                              elbo_best,
                              params_best,
                              bbvi_state.key,
                              bbvi_state.opt_state,
                              bbvi_state.params,
                              bbvi_state.elbo_train,
                              bbvi_state.elbo_val, 
                              bbvi_state.it), (curr_elbo, bbvi_state.grad)

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

        # Initialize the optimizer 
        opt_state = optimizer.init(self.init_var_params)

        # Split data into train and validation, shuffle first
        num_obs_train = round(self.num_obs*train_share)
        num_obs_val = self.num_obs - num_obs_train
        lost_obs = num_obs_train % batch_size
        if lost_obs > 0:
            batches = num_obs_train//batch_size + 1
        else:
            batches = num_obs_train//batch_size
        key, subkey = jax.random.split(key)
        rand_perm = jax.random.permutation(subkey, self.num_obs) 
        train_idx = rand_perm[:num_obs_train]   
        val_idx = rand_perm[num_obs_train:]
        data_train = jax.tree_map(lambda x: x[train_idx], self.data)
        data_val = jax.tree_map(lambda x: x[val_idx], self.data)

        epoch_state = EpochState(data_train,
                                 data_val, 
                                 num_obs_train, 
                                 num_obs_val,
                                 jnp.array(-jnp.inf),
                                 self.init_var_params,
                                 key,
                                 opt_state,
                                 self.init_var_params,
                                 jnp.zeros(epochs*batches),
                                 jnp.zeros(epochs*batches),
                                 0)
        
        elbo_epoch_hist = jnp.array([])
        grad_history = []
        dur = jnp.zeros(epochs // chunk_size)

        if self.verbose:
            print("Start optimization ...")

        for i in range(epochs // chunk_size):
            start = time.time()
            epoch_state, chunk = jscan(chunk_size, epoch_state)
            elbo_epoch_hist = jnp.append(elbo_epoch_hist, chunk[0], axis=0)
            grad_history.append(chunk[1])
            chunk_dur = time.time() - start
            elbo_delta = abs(elbo_epoch_hist[-1] - elbo_epoch_hist[-200]) if len(elbo_epoch_hist) > 200 else jnp.inf
            dur = dur.at[i].set(chunk_dur)
            if elbo_delta < threshold:
                break
        
        if self.verbose:
            print("Finished optimization.")

        self.elbo_hist["elbo_epoch"] = elbo_epoch_hist
        self.elbo_hist["epoch"] = jnp.arange(elbo_epoch_hist.shape[0])
        self.elbo_hist["grad"] = grad_history
        self.elbo_hist["elbo_full_train"] = epoch_state.elbo_train
        self.elbo_hist["elbo_full_val"] = epoch_state.elbo_val
        self.elbo_hist["iteration"] = jnp.arange(epoch_state.it)
        self.var_params = epoch_state.params_best
        self.it = epoch_state.it
        self.dur = dur
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
        Method to visualize the progression of the ELBO during the optimization
        considering only epochs.
        """

        plt.plot(self.elbo_hist["epoch"], self.elbo_hist["elbo_epoch"])
        plt.title("Progression of the ELBO during optimization")
        plt.xlabel("Epoch")
        plt.ylabel("ELBO")
        plt.show() 