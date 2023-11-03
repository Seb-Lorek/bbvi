"""
Multivariate normal distribution based on chol.
"""

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfjd

from typing import (
    Any,
    Tuple
)

def mvn_precision_chol_log_prob(x: jax.Array, 
                                loc: jax.Array, 
                                precision_matrix_chol: jax.Array) -> jax.Array:
    """
    Returns the log-density of a multivariate normal distribution parametrized by
    the Cholesky decomposition of the precision matrix.
    """

    z = (x - loc) @ precision_matrix_chol
    log_prob = jnp.sum(tfjd.Normal(loc=0., scale=1.).log_prob(z), axis=-1)
    log_det = jnp.sum(jnp.log(jnp.diag(precision_matrix_chol)))

    return log_prob + log_det

def mvn_precision_chol_sample(loc: jax.Array, 
                              precision_matrix_chol: jax.Array, 
                              noise: jax.Array) -> jax.Array:
    """
    Transform noise from standard Multivariate Normal distribution to a 
    Multivariate Normal distribution parametrized by the Cholesky decomposition
    of the precision matrix.
    """
    
    return jnp.add(loc, jax.lax.linalg.triangular_solve(precision_matrix_chol, 
                                                        noise, 
                                                        lower=True))

def mvn_sample_noise(key: jax.random.PRNGKey, shape: Tuple, S: int=1):
    """
    Sample from a standard normal distribution. 
    """

    sample_shape = (S, shape[-1]) if S > 1 else (shape[-1], )
    noise = jax.random.normal(key, sample_shape)

    return noise 
