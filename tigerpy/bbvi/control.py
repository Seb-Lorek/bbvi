"""
Control variate. 
"""

import jax
import jax.numpy as jnp

def quadratic_approx_fun(param, 
                         param_at, 
                         cv_vector,
                         cv_square_mat):
    """
    Quadratic function that approximates the reparameterization gradient estimator. 
    Allows for batching. 
    """

    centered = param - param_at
    lin = centered @ cv_vector
    centered = jnp.expand_dims(centered, axis=-2)
    centered_T = jnp.swapaxes(centered, -2, -1)
    cv_square_mat = jnp.diag(jnp.exp(cv_square_mat)**2)
    quad = jnp.squeeze(centered @ cv_square_mat @ centered_T, axis=(-2, -1))
    
    return lin + 1/2 * quad 

def quadratic_approx_mean(var_param_loc, 
                          var_param_cov, 
                          param_at, 
                          cv_vector, 
                          cv_square_mat):
    """
    Expectation w.r.t. the variational distribution of the quadratic function that
    approximates the reparameterization gradient estimator.
    """

    cv_square_mat = jnp.diag(jnp.exp(cv_square_mat)**2)
    lin = cv_vector.T @ (var_param_loc - param_at)
    trace = 1/2 * jnp.trace(cv_square_mat @ var_param_cov)
    term1 = var_param_loc.T @ cv_square_mat @ (var_param_loc - param_at)
    term2 = param_at.T @ cv_square_mat @ (param_at - var_param_loc)
    quad = 1/2 * (term1 + term2)
    
    return lin + trace + quad 

def calc_cv(var_param_loc, 
            var_param_cov, 
            param, 
            param_at, 
            cv_vector, 
            cv_square_mat):
    """
    Function to calculate the control variate.
    """

    m = quadratic_approx_mean(var_param_loc, 
                              var_param_cov, 
                              param_at, 
                              cv_vector, 
                              cv_square_mat)
    fun = quadratic_approx_fun(param, 
                               param_at, 
                               cv_vector, 
                               cv_square_mat)

    return m - jnp.mean(fun)

def calc_cv_weight(grad, 
                   grad_cv, 
                   old_weight, 
                   it, 
                   eps: float=1e-6):
    """
    Calculate the weight for the control variate. 
    """

    grad_vec = jnp.concatenate([d for d in grad.values()])
    grad_cv_vec = jnp.concatenate([d for d in grad_cv.values()])

    cov = grad_cv_vec @ grad_vec 
    var = grad_cv_vec @ grad_cv_vec + eps
    new_it = it + 1
    
    return it/new_it * old_weight - 1/new_it * cov/var

def calc_var_cv(grad, 
                grad_cv, 
                cv_weights):
    """
    Calculate the variance of the gradient.
    """

    norm = jax.tree_map(lambda x,y,z: x + z*y, grad, grad_cv, cv_weights)
    norm = jnp.concatenate([d for d in norm.values()])

    return norm.T @ norm