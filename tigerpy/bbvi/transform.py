"""
Transformation functions.
"""

from jax import jacfwd, vmap
import jax.numpy as jnp

# Define the log transformation function
def log_transform(x):
    return jnp.log(x)

# Define a function to compute the determinant of the Jacobian
def jac_determinant(f, x):
    jacobian = jacfwd(f)(x)
    return jnp.linalg.det(jacobian)

batched_jac_determinant = vmap(jac_determinant, (None, 0))
