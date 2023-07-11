"""
Utils.
"""

from jax import jit
import jax.numpy as jnp

from typing import Any

Array = Any

@jit
def dot(x: Array, y: Array) -> Array:
    """
    Function to calculate matix/dot products.

    Args:
        x (Array): Jax.numpy array. Column dimension must match with row dimension of y.
        y (Array): Jax.numpy array. Row dimension must match with column dimesion of x.

    Returns:
        Array: Jax.numpy array. The matrix/dot product.
    """
    return jnp.dot(x, y)

@jit
def quad_prod(A: Array, x: Array) -> Array:
    """
    Function to calculate quadratic forms.

    Args:
        A (Array): Jax.numpy array of shape (n,n).
        x (Array): 1-D Jax.numpy array of shape (n,).

    Returns:
        Array: Jax.numpy.array. Numeric results of the quadratic form.
    """
    return jnp.dot(x.T, jnp.dot(A, x))
