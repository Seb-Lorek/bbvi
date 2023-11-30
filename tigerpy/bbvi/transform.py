"""
Transformation functions.
"""

import jax
import jax.numpy as jnp
import numpy as np

# function to calculate the hessian of a function
def hessian(f, argnums):
  return jax.jacfwd(jax.grad(f, argnums=argnums), argnums=argnums)

# Use the unique definition of the Cholesky decomposition
def fill_lower_diag(log_vec_L: jax.Array, d: int) -> jax.Array:
    """
    Given a vector which parametrize a lower-traingular matrix, 
    reconstructs the matrix where the entries of the input vector
    are arranged in a row-column order.
    """

    mask = np.tri(d, dtype=bool)
    out = jnp.zeros((d, d), dtype=log_vec_L.dtype)

    return out.at[mask].set(log_vec_L)

def log_cholesky_parametrization(L: jax.Array, d: int) -> jax.Array:
    """
    Parametrize a lower-diagonal matrix through a vector with the
    elements in row-column order. The elements in the diagonal are
    moved in the log-space.
    """

    log_vec_L = L

    log_vec_L = jax.lax.fori_loop(
        0, 
        d, 
        lambda i, lvL: lvL.at[(i, i)].set(jnp.log(lvL[i][i])), 
        log_vec_L
    )

    return log_vec_L[jnp.tril_indices(d)]

def log_cholesky_parametrization_to_tril(log_vec_L: jax.Array, d: int) -> jax.Array:
    """
    Given a vector representing the log-cholesky parametrization,
    reconstructs the lower triangular matrix of the the cholesky
    decomposition.
    """

    log_cholesky_cov_tril = fill_lower_diag(log_vec_L, d)

    log_cholesky_cov_tril = jax.lax.fori_loop(
        0,
        d,
        lambda i, lcct: lcct.at[(i, i)].set(jnp.exp(lcct[i][i])),
        log_cholesky_cov_tril,
    )

    return log_cholesky_cov_tril

def cov_from_prec_chol(precision_matrix_chol: jax.Array) -> jax.Array:
    """
    Obtain the covariance matrix from the cholesky decomposition of the precision 
    matrix.
    """
    
    return jnp.linalg.inv(jnp.dot(precision_matrix_chol, precision_matrix_chol.T))

def solve_chol(chol_lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    """
    Solves a system of linear equations `chol_lhs @ x = rhs` for x,
    where `chol_lhs` is lower triangular matrix, by applying
    forward and backward substitution. Returns x.
    """

    tmp = jax.lax.linalg.triangular_solve(chol_lhs, rhs, left_side=True, lower=True)

    return jax.lax.linalg.triangular_solve(chol_lhs, tmp, lower=True)