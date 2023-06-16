"""
Utils.
"""

from jax import jit
import jax.numpy as jnp

from .model import(
    Array
)

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
