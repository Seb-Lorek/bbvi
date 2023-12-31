"""
Data-generating mechanisms.
"""

# Dependencies 
import numpy as np
import jax.numpy as jnp 
import pandas as pd 
import jax
import tensorflow_probability.substrates.jax.distributions as tfjd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import (
    Union, 
    Callable
)

"""
Simulate data.
"""

def sim_data_normal(range: Union[list, jax.Array],
                    n_obs: int,
                    fun_loc: Callable,
                    fun_scale: Union[Callable, None],
                    coef_loc: jax.Array,
                    coef_scale: jax.Array,
                    key: jax.Array) -> pd.DataFrame:
    
    key, *subkeys = jax.random.split(key, 3)
    x = jax.random.uniform(subkeys[0], shape=(n_obs,), minval=range[0], maxval=range[1])
    
    loc = fun_loc(x, coef_loc)

    if callable(fun_scale):
        log_scale = fun_scale(x, coef_scale)
        scale = jnp.exp(log_scale)
        # The shape is () sine scale already has corresponding shape 
        e = tfjd.Normal(loc=jnp.array(0.0), scale = scale).sample(sample_shape=(), seed=subkeys[1])
        print(e.shape)
    else:
        scale = coef_scale
        e = tfjd.Normal(loc=jnp.array(0.0), scale = scale).sample(sample_shape=x.shape, seed=subkeys[1])
    
    y = loc + e

    df = pd.DataFrame({
        "loc": loc,
        "scale": scale,
        "x": x,
        "y": y,
    })

    return df 

def sim_data_bernoulli(range: Union[list, jax.Array],
                       n_obs: int,
                       fun_logits: Callable,
                       coef_logits: jax.Array,
                       key: jax.Array) -> pd.DataFrame:
    
    key, *subkeys = jax.random.split(key, 3)
    x = jax.random.uniform(subkeys[0], shape=(n_obs,), minval=range[0], maxval=range[1])
    
    logits = fun_logits(x, coef_logits)
    probs = jnp.exp(logits)/(1 + jnp.exp(logits))

    # The array logits already has the corresponding shape 
    y = tfjd.Bernoulli(logits=logits).sample(sample_shape=(), seed=subkeys[1])

    df = pd.DataFrame({
        "logits": logits,
        "probs": probs,
        "x": x,
        "y": y,
    })

    return df

"""
Functions for the structural relationships in the linear predictor.
"""

def linear_fun(x, coef):
    y = coef[0] + coef[1]*x
    return y

def quadratic_fun(x, coef):
    y = coef[0] + coef[1]*x + coef[2]*x**2
    return y

def complex_fun(x, coef):
    y = coef[0] + coef[1]*jnp.sin(coef[2]*x)
    return y

"""
Functions to export.
"""

# All functions have the convention dist(response)_relationship(param1)_...

def normal_quadratic_const(n_obs, key):
    df = sim_data_normal([-3,3], 
                         n_obs, 
                         quadratic_fun,
                         None,
                         coef_loc=jnp.array([3.0, 0.2, -0.5]),
                         coef_scale=jnp.array(1.0),
                         key=key)
    
    data_dict = {"data": df,
                 "coef": {"loc": jnp.array([3.0, 0.2, -0.5]),
                          "scale": jnp.array(1.0)},
                 "rel": "quadratic, constant"
                 }

    return data_dict 

def bernoulli_linear(n_obs, key):
    df = sim_data_bernoulli([-3,3], 
                            n_obs, 
                            linear_fun,
                            coef_logits=jnp.array([1.0, 2.0]),
                            key=key)
    
    data_dict = {"data": df,
                 "coef": {"logits": jnp.array([1.0, 2.0])},
                 "rel": "linear"
                 }

    return data_dict

def normal_complex_const(n_obs, key):
    df = sim_data_normal([-10,10], 
                         n_obs, 
                         complex_fun,
                         None,
                         coef_loc=jnp.array([3.0, 1.75, 1.5]),
                         coef_scale=jnp.array(1.5),
                         key=key)
    
    data_dict = {"data": df,
                 "coef": {"loc": jnp.array([3.0, 1.75, 1.5]),
                          "scale": jnp.array(1.5)},
                 "rel": "complex, constant"
                 }

    return data_dict   

"""
Function to visualize the relationship.
"""

def plot_sim_data(df: pd.DataFrame, dist: str, savepath: None | str=None):
    
    if dist == "normal":
        df["upper"] = df["loc"] + 1.96*df["scale"]
        df["lower"] = df["loc"] - 1.96*df["scale"]

        sort_df = df.sort_values("x")

        sns.set_theme(style="whitegrid")
        fig = plt.figure(figsize=(8,6))
        plt.scatter(x=sort_df["x"], y=sort_df["y"], s=5)
        plt.plot(sort_df["x"], sort_df["loc"], color=sns.color_palette()[1])
        plt.plot(sort_df["x"], sort_df["upper"], linewidth=0.7, color=sns.color_palette()[1], linestyle='dashed')
        plt.plot(sort_df["x"], sort_df["lower"], linewidth=0.7, color=sns.color_palette()[1], linestyle='dashed')
        
        if savepath is not None:
            plt.savefig(savepath)
        else:
            plt.show()

    elif dist == "bernoulli":
        sort_df = df.sort_values("x")

        sns.set_theme(style="whitegrid")
        fig = plt.figure(figsize=(8,6))
        plt.scatter(x=sort_df["x"], y=sort_df["probs"], s=5)
        
        if savepath is not None:
            plt.savefig(savepath)
        else:
            plt.show()