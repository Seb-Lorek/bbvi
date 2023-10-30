"""
Multivariate normal distribution based on chol.
"""

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfjd

from typing import Any

Array = Any

def mvn_precision_chol_log_prob(x: Array, 
                                loc: Array, 
                                precision_matrix_chol: Array) -> Array:
    """
    Returns the log-density of a multivariate normal distribution parametrized by
    the Cholesky decomposition of the precision matrix.
    """

    z = (x - loc) @ precision_matrix_chol
    log_prob = jnp.sum(tfjd.Normal(loc=0., scale=1.).log_prob(z), axis=-1)
    log_det = jnp.sum(jnp.log(jnp.diag(precision_matrix_chol)))

    return log_prob + log_det

def mvn_precision_chol_sample(loc: Array, 
                              precision_matrix_chol: Array, 
                              key: Array, S: int=1) -> Array:
    """
    Samples from a Multivariate Normal distribution parametrized by
    the Cholesky decomposition of the precision matrix.
    """

    sample_shape = (S, precision_matrix_chol.shape[-1]) if S > 1 else (precision_matrix_chol.shape[-1], )
    
    z = jax.random.normal(key, sample_shape)
    
    return jnp.add(loc, jax.lax.linalg.triangular_solve(precision_matrix_chol, z, lower=True))

def mvn_trunc_precision_chol_log_prob(x: Array, 
                                      loc: Array, 
                                      precision_matrix_chol: Array, 
                                      min: float=0.0) -> Array:
    """
    Returns the log-density of a truncated multivariate normal distribution parametrized by
    the Cholesky decomposition of the precision matrix.
    """

    z = (x - loc) @ precision_matrix_chol
    log_prob = jnp.sum(tfjd.Normal(loc=0., scale=1.).log_prob(z), axis=-1)
    log_det = jnp.sum(jnp.log(jnp.diag(precision_matrix_chol)))
    adj = jnp.log(1 - tfjd.Normal(loc=0., scale=1.).cdf(min))

    return log_prob + log_det - adj

def mvn_trunc_precision_chol_sample(loc: Array, 
                                    precision_matrix_chol: Array,  
                                    key: Array, 
                                    min: float=0.0,
                                    S: int=1) -> Array:
    """
    Samples from a truncated Multivariate Normal distribution parametrized by
    the Cholesky decomposition of the precision matrix.
    """

    sample_shape = (S, precision_matrix_chol.shape[-1]) if S > 1 else (precision_matrix_chol.shape[-1], )
    sample_num = sample_shape[-2]*sample_shape[-1] if S > 1 else sample_shape[-1]
    samples = jnp.zeros(sample_num)
    loop_carry = {"samples": samples, "key": key}
    
    def reject_samp(loop_carry):
        key, subkey = jax.random.split(loop_carry["key"])
        s = jax.random.normal(subkey, (sample_num,))
        mask = s > min
        accepted = jnp.where(mask, s, 0.0)
        samples = jnp.where(accepted>min, accepted, loop_carry["samples"])
        return {"samples": samples, "key": key} 
    
    carry = jax.lax.while_loop(
        lambda loop_carry: jnp.count_nonzero(loop_carry["samples"]) < sample_num,
        reject_samp,
        loop_carry
    )

    z_trunc = carry["samples"]
    z_trunc = jnp.reshape(z_trunc, sample_shape)

    return jnp.add(loc, jax.lax.linalg.triangular_solve(precision_matrix_chol, z_trunc, lower=True))


def solve_chol(chol_lhs: Array, rhs: Array) -> Array:
    """
    Solves a system of linear equations `chol_lhs @ x = rhs` for x,
    where `chol_lhs` is lower triangular matrix, by applying
    forward and backward substitution. Returns x.
    """

    tmp = jax.lax.linalg.triangular_solve(chol_lhs, rhs, left_side=True, lower=True)

    return jax.lax.linalg.triangular_solve(chol_lhs, tmp, lower=True)
