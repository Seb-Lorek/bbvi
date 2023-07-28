"""
Multivariate Normal Distribution based on Chol.
"""

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfjd

from typing import Any

Array = Any

def mvn_precision_chol_log_prob(x: Array, loc: Array, precision_matrix_chol: Array) -> Array:
    """
    Returns the log-density of a multivariate normal distribution parametrized by
    the Cholesky decomposition of the precision matrix.
    """
    z = (x - loc) @ precision_matrix_chol
    log_prob = jnp.sum(tfjd.Normal(loc=0., scale=1.).log_prob(z))
    log_det = jnp.sum(jnp.log(jnp.diag(precision_matrix_chol)))

    return log_prob + log_det


def mvn_precision_chol_sample(loc: Array, precision_matrix_chol: Array, key: Array, S: int=1) -> Array:
    """
    Samples from a Multivariate Normal distribution parametrized by
    the Cholesky decomposition of the precision matrix.
    """
    sample_shape = (S, precision_matrix_chol.shape[-1]) if S > 1 else (precision_matrix_chol.shape[-1], )

    z = jax.random.normal(key, sample_shape)

    return jnp.add(loc, jax.lax.linalg.triangular_solve(precision_matrix_chol, z, lower=True))

def solve_chol(chol_lhs: Array, rhs: Array) -> Array:
    """
    Solves a system of linear equations `chol_lhs @ x = rhs` for x,
    where `chol_lhs` is lower triangular matrix, by applying
    forward and backward substitution. Returns x.
    """

    tmp = jax.lax.linalg.triangular_solve(chol_lhs, rhs, left_side=True, lower=True)

    return jax.lax.linalg.triangular_solve(chol_lhs, tmp, lower=True)
