"""
Helper functions that calculate quantities form the graph.
"""

import jax 
import jax.numpy as jnp 

import networkx as nx

from typing import (
    Dict, 
    Callable,
    Tuple
)

from ..model.model import (
    Distribution,
    Array,
    Any
)

from ..distributions.mvn import (
    mvn_precision_chol_log_prob,
    mvn_precision_chol_sample,
    mvn_sample_noise
)

from ..distributions.mvn_log import (
    mvn_log_precision_chol_log_prob,
    mvn_log_precision_chol_sample
)

from .transform import (
    log_cholesky_parametrization_to_tril
)

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

def gen_noise(var_params, 
              num_var_samples, 
              key) -> Dict:
    samples_noise = {}
    key, *subkeys = jax.random.split(key, len(var_params)+1)
    for i, kw in enumerate(var_params.keys()):
        samples_noise[kw] = mvn_sample_noise(key=subkeys[i],
                                             shape= var_params[kw]["loc"].shape,
                                             S=num_var_samples)
    return samples_noise