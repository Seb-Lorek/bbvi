"""
Initialize the variational parameters.
"""

import jax
import jax.numpy as jnp

import tensorflow_probability.substrates.jax.distributions as tfjd

from typing import (
    Any, 
    Union,
    Dict,
)

from .transform import (
    log_cholesky_parametrization
)

def set_init_var_params(attr: Any, 
                        param_response: str=None, 
                        key: Union[jax.random.PRNGKey, None]=None, 
                        scale_prec: float=10.0) -> Dict:
    """
    Function to initialize the variational parameters.

    Args:
        attr (Any): Attributes of a parameter node in the DAG.
        param_response (str, optional): String that defines the name of the response parameter 
        to which the parameter block belongs to. Defaults to None.
        key (Union[jax.random.PRNGKey, None], optional): Indicate if initalization should apply jitter or not. Defaults to None.
        scale_prec (float, optional): Diagonal of the precision matrix from the variational distribution for location parameters. Defaults to 10.0.

    Returns:
        Dict: The initial interal variational parameters.
    """

    if not key == None:
        key, *subkeys = jax.random.split(key, 3)
        if param_response == "scale":
            if attr["param_space"] is None:
                loc = tfjd.Normal(loc=attr["value"], scale=jnp.array([0.01])).sample(sample_shape=(), seed=subkeys[0])
                prec = tfjd.Normal(loc=jnp.repeat(scale_prec, attr["dim"]), scale=jnp.array([0.01])).sample(sample_shape=(), seed=subkeys[1])
                lower_tri = jnp.tril(jnp.diag(prec))
                log_cholesky_prec = log_cholesky_parametrization(lower_tri, d=lower_tri.shape[0])
            elif attr["param_space"] == "positive":
                loc = tfjd.Normal(loc=jnp.log(attr["value"]), scale=jnp.array([0.01])).sample(sample_shape=(), seed=subkeys[0])
                prec = tfjd.Normal(loc=jnp.repeat(scale_prec, attr["dim"]), scale=jnp.array([0.01])).sample(sample_shape=(), seed=subkeys[1])
                lower_tri = jnp.tril(jnp.diag(prec))
                log_cholesky_prec = log_cholesky_parametrization(lower_tri, d=lower_tri.shape[0])
        else:
            if attr["param_space"] is None:
                loc = tfjd.Normal(loc=attr["value"], scale=jnp.array([0.01])).sample(sample_shape=(), seed=subkeys[0])
                prec = tfjd.Normal(loc=jnp.repeat(1.0, attr["dim"]), scale=jnp.array([0.01])).sample(sample_shape=(), seed=subkeys[1])
                lower_tri = jnp.tril(jnp.diag(prec))
                log_cholesky_prec = log_cholesky_parametrization(lower_tri, d=lower_tri.shape[0])
            elif attr["param_space"] == "positive":
                loc = tfjd.Normal(loc=jnp.log(attr["value"]), scale=jnp.array([0.01])).sample(sample_shape=(), seed=subkeys[0])
                prec = tfjd.Normal(loc=jnp.repeat(1.0, attr["dim"]), scale=jnp.array([0.01])).sample(sample_shape=(), seed=subkeys[1])
                lower_tri = jnp.tril(jnp.diag(prec))
                log_cholesky_prec = log_cholesky_parametrization(lower_tri, d=lower_tri.shape[0])
    else:
        if param_response == "scale":
            if attr["param_space"] is None:
                loc = attr["value"]
                prec = jnp.repeat(scale_prec, attr["dim"])
                lower_tri = jnp.tril(jnp.diag(prec))
                log_cholesky_prec = log_cholesky_parametrization(lower_tri, d=lower_tri.shape[0])
            elif attr["param_space"] == "positive":
                loc = jnp.log(attr["value"])
                prec = jnp.repeat(scale_prec, attr["dim"])
                lower_tri = jnp.tril(jnp.diag(prec))
                log_cholesky_prec = log_cholesky_parametrization(lower_tri, d=lower_tri.shape[0])
        else:
            if attr["param_space"] is None:
                loc = attr["value"]
                prec = jnp.repeat(1.0, attr["dim"])
                lower_tri = jnp.tril(jnp.diag(prec))
                log_cholesky_prec = log_cholesky_parametrization(lower_tri, d=lower_tri.shape[0])
            elif attr["param_space"] == "positive":
                loc = jnp.log(attr["value"])
                prec = jnp.repeat(1.0, attr["dim"])
                lower_tri = jnp.tril(jnp.diag(prec))
                log_cholesky_prec = log_cholesky_parametrization(lower_tri, d=lower_tri.shape[0])
    
    return {"loc": loc, "log_cholesky_prec": log_cholesky_prec}