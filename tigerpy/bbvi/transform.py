"""
Transformation functions.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jacfwd, vmap

from typing import (
    Any
)

Array = Any

# Define the log transformation function
def log_transform(x):
    return jnp.log(x)

# Define the exp transformation function
def exp_transform(x):
    return jnp.exp(x)

# Define a function to compute the determinant of the Jacobian
def jac_determinant(f, x):
    jacobian = jacfwd(f)(x)
    return jnp.linalg.det(jacobian)

batched_jac_determinant = vmap(jac_determinant, (None, 0))

# Note: while being general this is not efficient for complicated Jacobians
# Leave it for the moment at this implementation

# Use the unique definition of the Cholesky decomposition
def fill_lower_diag(log_vec_L: Array, d: int) -> Array:
    """
    Given a vector which parametrize a lower-traingular matrix, 
    reconstructs the matrix where the entries of the input vector
    are arranged in a row-column order.
    """

    mask = np.tri(d, dtype=bool)
    out = jnp.zeros((d, d), dtype=log_vec_L.dtype)

    return out.at[mask].set(log_vec_L)

def log_cholesky_parametrization(L: Array, d: int) -> Array:
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

def log_cholesky_parametrization_to_tril(log_vec_L: Array, d: int) -> Array:
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