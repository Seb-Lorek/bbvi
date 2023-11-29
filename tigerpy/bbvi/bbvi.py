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
import seaborn as sns
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

from ..distributions.mvn_log import (
    mvn_log_mean,
    mvn_log_cov
)

from .transform import (
    log_cholesky_parametrization,
    log_cholesky_parametrization_to_tril,
    cov_from_prec_chol,
    hessian
)

from .init_bbvi import (
    set_init_var_params,
    add_jitter
)

from .calc import(
    calc_lpred,
    calc_scaled_loglik,
    init_dist,
    logprior,
    loglik,
    gen_noise,
    neg_entropy_unconstr,
    neg_entropy_posconstr
)

class MapState(NamedTuple):
    data_train: dict
    data_val: dict
    batches: list
    num_obs_train: int
    num_obs_val: int 
    lprob_best: jax.Array
    params_best: dict
    opt_state: Any
    params: dict
    lprob_train: jax.Array
    lprob_val: jax.Array
    it: int

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
                 jitter: float=1e-3,
                 pre_train: bool=True,
                 loc_prec: float=1.0,
                 scale_prec: float=2.0, 
                 verbose=True) -> None:
        self.graph = copy.deepcopy(graph)
        self.digraph = self.graph.digraph
        self.model = self.graph.model
        self.num_obs = self.model.num_obs
        self.data = {}
        self.jitter_init = jitter_init
        self.jitter = jitter
        self.pre_train = pre_train
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
        Method to set the internal variational parameters, to initial values of the graph and sepcify diagonal of the prec.
        """
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

    def mc_logprob(self, 
                   samples: Dict,
                   data: Dict,
                   num_obs: int) -> jax.Array:
        """
        Calculate the monte carlo integral of the joint log-probability of the model, can handle batches of samples 
        from the variational distribution.
        Args:
            samples (Dict): The parameters of the model, can be batches.
            data (Dict): A subsample of the data.
            num_obs (int): Number of observations in the model.
        Returns:
            jax.Array: Monte Carlo integral for the noisy log-probability of the model. We only use a subsample
            of the data.
        """
        # Safe intermediary computations in the model_state
        model_state = {}
        # Safe all log priors 
        # Maybe explicity define shape log-priors via num_var_samples
        # However gets recasted correctly 
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
                lpred_val = calc_lpred(design_matrix, 
                                       params,
                                       attr["bijector"])
                model_state[node] = lpred_val   
            # Strong node          
            elif node_type == "strong":
                if self.digraph.nodes[node]["input_fixed"]:
                    log_prior = logprior(attr["dist"], 
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
                    dist = init_dist(dist=attr["dist"], 
                                     params=params)
                    log_prior = logprior(dist, 
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
                dist = init_dist(dist=attr["dist"],
                                 params=params)
                # print("Response dist:", init_dist)
                # Calculate the log_lik
                log_lik = loglik(dist, 
                                 data[node])
                # print("Log-lik:", log_lik, log_lik.shape)
                # print("Mean log-lik", jnp.mean(log_lik, axis=-1), jnp.mean(log_lik, axis=-1).shape)
                # print("Scaled log-lik", num_obs * jnp.mean(log_lik, axis=-1), jnp.mean(log_lik, axis=-1).shape)
                # calculate the scaled log-likelihood
                scaled_log_lik = calc_scaled_loglik(log_lik,
                                                    num_obs)

        # print("Scaled log-lik", jnp.mean(scaled_log_lik), jnp.mean(scaled_log_lik).shape)
        # print("Log-priors", jnp.mean(log_priors), jnp.mean(log_priors).shape)
        return jnp.mean(scaled_log_lik + log_priors)

    # Update docstrings
    def lower_bound(self, 
                    var_params: Dict,
                    data: Dict,
                    num_obs: int,
                    samples_noise: Dict) -> Array:

        samples_params = {}
        total_neg_entropy = jnp.array([])

        for kw in var_params.keys():
            if self.digraph.nodes[kw]["attr"]["param_space"] is None:
                samples_params, neg_entropy = neg_entropy_unconstr(var_params, 
                                                                   samples_params,
                                                                   samples_noise, 
                                                                   kw)                                                                           
                total_neg_entropy = jnp.append(total_neg_entropy, neg_entropy, axis=0)
            elif self.digraph.nodes[kw]["attr"]["param_space"] == "positive":
                samples_params, neg_entropy = neg_entropy_posconstr(var_params, 
                                                                    samples_params,
                                                                    samples_noise, 
                                                                    kw)
                total_neg_entropy = jnp.append(total_neg_entropy, neg_entropy, axis=0)

        # print(samples["tau2"], samples["eta2"])
        # print(total_neg_entropy)

        mc_log_prob = self.mc_logprob(samples_params, 
                                      data, 
                                      num_obs)
        elbo = mc_log_prob - jnp.sum(total_neg_entropy)
        
        return - elbo

    # Update Docstrings 
    def run_bbvi(self,
                 key: jax.random.PRNGKey,
                 learning_rate: Union[Any, float]=1e-3,
                 pre_train_learning_rate: float=1e-2,
                 grad_clip: float=1,
                 threshold: float=1e-2,
                 batch_size: int=64,
                 pre_train_batch_size: int=64,
                 train_share: float=0.8,
                 num_var_samples=32,
                 chunk_size: int=1,
                 epochs: int=500) -> tuple:
        """
        Method to run the stochastic gradient optimization algorithm. 
        The implementation uses Adam.
        """
        key, subkey = jax.random.split(key)
        self.post_samples_key = subkey

        if type(learning_rate) is float:
            optimizer = optax.chain(optax.clip(grad_clip), 
                                    optax.adam(learning_rate=learning_rate))
        else:
            optimizer = optax.clip(optax.clip(grad_clip), 
                                   optax.adamw(learning_rate=learning_rate))
            
        def set_init_map(data_train: dict,
                         data_val: dict,
                         init_var_params: dict, 
                         pre_train_batch_size: int,
                         pre_train_learning_rate: int,
                         grad_clip: float,
                         key: jax.Array) -> Dict:
            init_params = {kw: value["loc"] for kw, value in init_var_params.items()}
            num_obs_train = data_train["response"].shape[0]
            num_obs_val = data_val["response"].shape[0]
            log_prob = self.mc_logprob(init_params, 
                                       data_val,  
                                       num_obs_val)
            optimizer = optax.chain(optax.clip(grad_clip), 
                                    optax.adam(learning_rate=pre_train_learning_rate))
            opt_state = optimizer.init(init_params)
            log_probs = jnp.array([log_prob])

            # incluce batches into MapState
            map_state = MapState(data_train, 
                                 data_val, 
                                 num_obs_train,
                                 num_obs_val, 
                                 jnp.array(-jnp.inf),
                                 init_params, 
                                 opt_state, 
                                 init_params,
                                 jnp.zeros(pre_train_batch_size), 
                                 jnp.zeros(pre_train_batch_size), 
                                 0)
            
            delta = jnp.inf
            while abs(delta) > 0.5:
                key, *subkeys = jax.random.split(key, 3)

                # Split data into train batches 
                train_idx = jax.random.permutation(subkeys[0], num_obs_train)
                # Note: batch_size is a global variable
                num_batches = num_obs_train // pre_train_batch_size 
                lost_obs = num_obs_train % pre_train_batch_size
                if lost_obs > 0:
                    add_idx = jax.random.choice(subkeys[1], train_idx[:-lost_obs], (pre_train_batch_size-lost_obs,), replace=False)
                    train_idx = jnp.append(train_idx, add_idx) 
                    batches = jnp.split(train_idx, (num_batches+1))
                else:
                    batches = jnp.split(train_idx, num_batches)

                map_state = jax.lax.fori_loop(0, 
                                              len(batches), 
                                              map_body, 
                                              map_state)
                log_probs = jnp.append(log_probs, map_state.lprob_best)
                delta = log_probs[map_state.it] - log_probs[map_state.it - 1]
                print(abs(delta))
                
            print("Best log-prob:", map_state.lprob_best)
            print("Parameters:", map_state.params_best)

            # Get the MAP parameters 
            init_var_params = {}
            for kw, value in map_state.params_best.items():
                init_var_params[kw]["loc"] = value
                H = hessian(self.mc_logprob, argnums=0)(map_state.params_best, 
                                                        map_state.data_val, 
                                                        map_state.num_obs_val)
                L = jnp.linalg.cholesky(H)
                init_var_params[kw]["log_cholesky_prec"] = log_cholesky_parametrization(L, d=L.shape[0])
                
            return init_var_params
        
        def map_body(idx, map_state):
            funcs = [lambda i=i: map_state.batches[i] for i in range(len(map_state.batches))]
            batch_idx = jax.lax.switch(idx, funcs)
            batch_data = jax.tree_map(lambda x: x[batch_idx], map_state.data_train)
            lprob_grad_value = jax.value_and_grad(self.mc_logprob)
            lprob_train, grad = lprob_grad_value(map_state.params,
                                                 batch_data,
                                                 map_state.num_obs_train)
            neg_grad = jax.tree_map(lambda x: x*(-1), grad)
            updates, opt_state = optimizer.update(neg_grad, 
                                                  map_state.opt_state, 
                                                  map_state.params)
            params_new = optax.apply_updates(map_state.params,
                                             updates)
            
            lprob_val = self.mc_logprob(params_new, 
                                         data_val, 
                                         map_state.num_obs_val)
            
            lprob_train = map_state.lprob_train.at[map_state.it].set(lprob_train)
            lprob_val = map_state.lprob_val.at[map_state.it].set(lprob_val)

            # Choose max logprob here and pass to MapState 
            def true_fn(params, lprob, params_best, lprob_best):
                return params, lprob
            
            def false_fn(params, lprob, params_best, lprob_best):
                return params_best, lprob_best
            
            params_best, lprob_best = jax.lax.cond(lprob_val > map_state.lprob_best,
                                                  true_fn,
                                                  false_fn,
                                                  map_state.params,
                                                  lprob_val,
                                                  map_state.params_best, 
                                                  map_state.lprob_best)
            return MapState(map_state.data_train, 
                            map_state.data_val, 
                            map_state.num_obs_train, 
                            map_state.num_obs_val, 
                            lprob_best, 
                            params_best, 
                            opt_state, 
                            params_new, 
                            lprob_train, 
                            lprob_val, 
                            map_state.it + 1)
        
        def bbvi_body(idx, bbvi_state):
            
            key, subkey = jax.random.split(bbvi_state.key)
            funcs = [lambda i=i: bbvi_state.batches[i] for i in range(len(bbvi_state.batches))]
            batch_idx = jax.lax.switch(idx, funcs)
            batch_data = jax.tree_map(lambda x: x[batch_idx], bbvi_state.data_train)
            
            # Generate noise samples 
            samples_noise = gen_noise(bbvi_state.params,    
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

        # Data preprocessing 
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

        # Set the initial parameter values
        if self.pre_train:
            key, subkey = jax.random.split(key)
            if self.verbose:
                print("Start pre training ...")
            self.init_var_params = set_init_map(data_train,
                                                data_val,
                                                self.init_var_params, 
                                                pre_train_batch_size,
                                                pre_train_learning_rate,
                                                grad_clip,
                                                subkey)
            if self.verbose:
                print("Finished pre training")

        if self.jitter_init:
            key, subkey = jax.random.split(key)
            self.init_var_params = add_jitter(self.init_var_params,
                                              self.jitter, 
                                              subkey)   

        # Initialize the optimizer 
        opt_state = optimizer.init(self.init_var_params)

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
            print("Finished optimization")

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
        
        sns.set_theme(style="whitegrid")

        plt.plot(self.elbo_hist["epoch"], self.elbo_hist["elbo_epoch"])
        plt.title("Progression of the ELBO during optimization")
        plt.xlabel("Epoch")
        plt.ylabel("ELBO")
        plt.show() 

        sns.reset_orig()

    def get_posterior_samples(self, 
                              sample_shape: Tuple) -> Dict[str, jax.Array]:
        post_sample = {}
        key = self.post_samples_key
        for kw in self.var_params:
            key, subkey = jax.random.split(key)
            if self.digraph.nodes[kw]["attr"]["param_space"] is None:
                loc = self.var_params[kw]["loc"]
                lower_tri = log_cholesky_parametrization_to_tril(self.var_params[kw]["log_cholesky_prec"], 
                                                                 d=loc.shape[0])
                cov = cov_from_prec_chol(lower_tri)
                dist = tfjd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=cov)
                samples = dist.sample(sample_shape=sample_shape, seed=subkey)
                post_sample[kw] = samples
            elif self.digraph.nodes[kw]["attr"]["param_space"] == "positive":
                loc = self.var_params[kw]["loc"]
                lower_tri = log_cholesky_parametrization_to_tril(self.var_params[kw]["log_cholesky_prec"], 
                                                                 d=loc.shape[0])
                cov = cov_from_prec_chol(lower_tri)
                dist = tfjd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=cov)
                samples = dist.sample(sample_shape=sample_shape, seed=subkey)
                post_sample[kw] = jnp.exp(samples)

        return post_sample 