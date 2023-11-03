"""
Multivariate log normal distribution.
"""

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfjd

from typing import (
    Any,
    Tuple
)

def mvn_log_precision_chol_log_prob(x: jax.Array, 
                                    loc: jax.Array, 
                                    precision_matrix_chol: jax.Array) -> jax.Array:
    """
    Returns the log-density of a multivariate log normal distribution parametrized by
    the Cholesky decomposition of the precision matrix of its corresponding multivariate normal
    distribution.
    """

    z = (jnp.log(x) - loc) @ precision_matrix_chol
    log_prob = jnp.sum(tfjd.Normal(loc=0., scale=1.).log_prob(z), axis=-1)
    log_det = jnp.sum(jnp.log(jnp.diag(precision_matrix_chol)))
    log_det_jacob = - jnp.sum(jnp.log(x), axis=-1)

    return log_prob + log_det + log_det_jacob

def mvn_log_precision_chol_sample(loc: jax.Array, 
                                  precision_matrix_chol: jax.Array, 
                                  noise: jax.Array) -> jax.Array:
    """
    Transform noise from standard multivariate normal distribution to a 
    multivariate log normal distribution parametrized by the Cholesky decomposition
    of the precision matrix of its corresponding multivariate normal distribution.
    """
    
    return jnp.exp(jnp.add(loc, jax.lax.linalg.triangular_solve(precision_matrix_chol, 
                                                                noise, 
                                                                lower=True)))

def mvn_log_mean(loc_norm: jax.Array, 
                 cov_norm: jax.Array) -> jax.Array:
    """
    Obtain the mean of the multivariate log normal distribution from its corresponding
    multivariate normal distribution.
    """
    
    return jnp.exp(loc_norm + 1/2 * jnp.diag(cov_norm))

def mvn_log_cov(loc_norm: jax.Array, 
                cov_norm: jax.Array) -> jax.Array:
    """
    Calculate the covariance matrix of the multivariate log normal distribution 
    from its corresponding multivariate normal distribution.
    """
    
    d = cov_norm.shape[0]
    diag_cov_norm = jnp.diag(cov_norm)
    m = jnp.outer(loc_norm, jnp.ones(d)) + jnp.outer(jnp.ones(d), loc_norm)
    c = jnp.outer(diag_cov_norm, jnp.ones(d)) + jnp.outer(jnp.ones(d), diag_cov_norm)
    term1 = jnp.exp(m + c/2)
    term2 = jnp.exp(cov_norm) - 1

    return term1 * term2
