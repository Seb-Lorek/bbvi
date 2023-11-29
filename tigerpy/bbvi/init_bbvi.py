"""
Initialize the variational parameters.
"""

import jax
import jax.numpy as jnp

from scipy.linalg import block_diag

import tensorflow_probability.substrates.jax.distributions as tfjd

import networkx as nx

from typing import (
    Any, 
    Union,
    Dict,
)

from .transform import (
    log_cholesky_parametrization
)

from ..distributions.mvn_degen import (
    MultivariateNormalDegenerate
)

def set_init_var_params(attr: Any, 
                        loc_prec: float,
                        scale_prec: float) -> Dict[str, jax.Array]:
    """
    Function to initialize the variational parameters.

    Args:
        attr (Any): Attributes of a parameter node in the DAG.
        to which the parameter block belongs to. Defaults to None.
        loc_prec (float): Diagonal of the precision matrix from the
        variational distribution for location parameters.
        scale_prec (float): Diagonal of the precision matrix from the 
        variational distribution for location parameters.

    Returns:
        Dict: The initial interal variational parameters.
    """

    if attr["param_response"] == "scale":
        if attr["param_space"] is None:
            loc = attr["value"]
            log_cholesky_prec = set_log_cholesky_prec(scale_prec, attr["dim"])
        elif attr["param_space"] == "positive":
            loc = jnp.log(attr["value"])
            log_cholesky_prec = set_log_cholesky_prec(scale_prec, attr["dim"])
    else:
        if attr["param_space"] is None:
            loc = attr["value"]
            log_cholesky_prec = set_log_cholesky_prec(loc_prec, attr["dim"])
        elif attr["param_space"] == "positive":
            loc = jnp.log(attr["value"])
            log_cholesky_prec = set_log_cholesky_prec(loc_prec, attr["dim"])
    
    return {"loc": loc, "log_cholesky_prec": log_cholesky_prec}

def set_init_loc_scale(digraph: nx.DiGraph,
                       prob_traversal_order: list,
                       loc_prec: float,
                       scale_prec: float) -> Dict[str, Dict[str, jax.Array]]:
    """
    Function to initialize the variational parameters for location scale regression.

    Args:
        digraph (nx.DiGraph): A networkx directed graph.
        prob_traversal_order (list):
        loc_prec (float): Diagonal of the precision matrix from the
        variational distribution for location parameters. 
        scale_prec (float): Diagonal of the precision matrix from the 
        variational distribution for location parameters.

    Returns:
        Dict[str, Dict[str, jax.Array]]: Nested dict with the initial 
        variational parameters
    """
    init_var_params = {}

    parents = list(digraph.predecessors("response"))
    y = digraph.nodes["response"]["attr"]["value"]

    for parent in parents:
        if parent == "loc":
            input = digraph.nodes[parent]["input"]
            design_mat_loc = input["fixed"]

            pen_loc = set_pen(digraph,
                              parent)
            coefs_loc = calc_coefs(design_mat_loc,
                                   y,
                                   pen_loc)
            parents_loc = list(digraph.predecessors(parent))
            for parent_loc in parents_loc:
                if digraph.nodes[parent_loc]["node_type"] != "strong":
                    parents_loc.remove(parent_loc)   
        elif parent == "scale":
            s = jnp.log(jnp.abs(y - design_mat_loc @ coefs_loc)) + 0.635
            input = digraph.nodes[parent]["input"]
            design_mat_scale = input["fixed"]

            pen_scale = set_pen(digraph,
                                parent)
            coefs_scale = calc_coefs(design_mat_scale,
                                     s,
                                     pen_scale)
            
            sig = jnp.exp(design_mat_scale @ coefs_scale)
            weights = jnp.diag(sig)
            
            coefs_loc = calc_coefs_weight(design_mat_loc,
                                          y,
                                          pen_scale,
                                          weights)
            
            parents_scale = list(digraph.predecessors(parent))
            for parent_scale in parents_scale:
                if digraph.nodes[parent_scale]["node_type"] != "strong":
                    parents_scale.remove(parent_scale) 
    
    if "loc" in parents:
        last = 0
        for parent_loc in parents_loc:
            init_var_params[parent_loc] = {}
            attr = digraph.nodes[parent_loc]["attr"]
            init_var_params[parent_loc]["loc"] = coefs_loc[last:(last+attr["dim"])]
            init_var_params[parent_loc]["log_cholesky_prec"] = set_log_cholesky_prec(loc_prec, attr["dim"])
            last += attr["dim"]
        last = 0
    if "scale" in parents:
        for parent_scale in parents_scale:
            init_var_params[parent_scale] = {}
            attr = digraph.nodes[parent_scale]["attr"]
            init_var_params[parent_scale]["loc"] = coefs_scale[last:(last+attr["dim"])]
            init_var_params[parent_scale]["log_cholesky_prec"] = set_log_cholesky_prec(scale_prec, attr["dim"])
            last += attr["dim"]

    for node in prob_traversal_order:
        if node not in list(init_var_params.keys()) and node != "response":
            attr = digraph.nodes[node]["attr"]
            init_var_params[node] = set_init_var_params(attr,
                                                        loc_prec,
                                                        scale_prec)

    return init_var_params

def set_pen(digraph: nx.DiGraph,
            predictor: str) -> jax.Array:
    """
    Function to create a penalty matrix for the initialization of location scale regression models.

    Args:
        digraph (nx.DiGraph): A networkx directed graph graph.
        predictor (str): Predictor of the regression model either location or scale.

    Returns:
        jax.Array: A block matrix containing the penalty matrix for the initialization.
    """

    parents = list(digraph.predecessors(predictor))
    pen_store = []
    for parent in parents:
        node_type = digraph.nodes[parent]["node_type"]
        if node_type == "strong":
            attr = digraph.nodes[parent]["attr"]
            input_pass = digraph.nodes[parent]["input"]
            if attr["dist"] is MultivariateNormalDegenerate:
                pen_store.append(input_pass["pen"])
            else:
                pen = jnp.zeros((attr["dim"], attr["dim"]))
                pen_store.append(pen)

    return jnp.asarray(block_diag(*(pen_store)))

def calc_coefs(design_mat: jax.Array, 
               y: jax.Array,
               pen: jax.Array) -> jax.Array:
    """
    Function to calculate the coefficients via PLS (penalized least squares).

    Args:
        design_mat (jax.Array): The design matrix of the predictor.
        y (jax.Array): The response vector.
        pen (jax.Array): The penalty matrix.

    Returns:
        jax.Array: The coefficients obtained from PLS.
    """

    coefs = jnp.linalg.inv(design_mat.T @ design_mat + pen) @ design_mat.T @ y
    
    return coefs

def calc_coefs_weight(design_mat: jax.Array, 
                      y: jax.Array,
                      pen: jax.Array,
                      weights = jax.Array) -> jax.Array:
    """
    Function to calculate the coefficients via weighted PLS (penalized least squares).


    Args:
        design_mat (jax.Array): The design matrix of the predictor.
        y (jax.Array): The response vector.
        pen (jax.Array): The penalty matrix.
        weights (jax.Array): The weight matrix. Defaults to jax.Array.

    Returns:
        jax.Array:  The coefficients obtained from weighted PLS.
    """

    coefs = jnp.linalg.inv(design_mat.T @ weights @ design_mat + pen) @ design_mat.T @ weights @ y
    
    return coefs

def set_log_cholesky_prec(prec_diag: float, 
                          dim: int) -> jax.Array:
    """
    Function to create a vector of the lower triangular of the Cholesky 
    decomposition of the precision matrix, where all diagonal elements are logarithmized.

    Args:
        prec_diag (float): The diagonal specification of the precision matrix.
        dim (int): The dimension of the parameter.

    Returns:
        jax.Array: The vector of the lower triangular of the Cholesky 
        decomposition of the precision matrix, where all diagonal elements
        are logarithmized.
    """

    prec = jnp.repeat(prec_diag, dim)
    lower_tri = jnp.tril(jnp.diag(prec))
    log_cholesky_prec = log_cholesky_parametrization(lower_tri, d=lower_tri.shape[0])
    
    return log_cholesky_prec

def add_jitter(params: Dict,
               jitter: float, 
               key: jax.random.PRNGKey) -> Dict[str, Dict[str, jax.Array]]:
    """
    Function to add jitter to the initial variational parameters.

    Args:
        params (Dict): The variational parameters in a nested dictionary.
        jitter (float): The scale of the jitter that is added to initializations.
        key (jax.random.PRNGKey): A jax.random.PRNGKey

    Returns:
        Dict[str, Dict[str, jax.Array]]: A nested dictionary with the initial 
        variational parameters with additional noise.
    """
    
    jitter_store = {}

    for kw in params.keys():
        jitter_store[kw] = {}
        key, *subkeys = jax.random.split(key, 3)
        loc_dim = jnp.shape(params[kw]["loc"])
        scale_dim = jnp.shape(params[kw]["log_cholesky_prec"])
        jitter_store[kw]["loc"] = tfjd.Normal(loc=0.0, scale=jitter).sample(sample_shape=loc_dim, seed=subkeys[0])
        jitter_store[kw]["log_cholesky_prec"] = tfjd.Normal(loc=0.0, scale=jitter).sample(sample_shape=scale_dim, seed=subkeys[1])

    params_jitter = jax.tree_map(lambda x, j: x + j, params, jitter_store)

    return params_jitter    