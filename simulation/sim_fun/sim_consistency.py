"""
Simulation for convergence in mean-square. 
"""

from . import sim_data

import itertools
import sys
import os
import time 
from typing import (
    Union,
    Tuple,
    List,
    Dict,
    Any
)
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfjd

# Set path such that interpreter finds tigerpy
# Assuming we execute the file from the top project directory 
sys.path.append(os.getcwd())

import tigerpy.model as tiger
import tigerpy.bbvi as bbvi

# Function to execute the simulation study 
def sim_fun(n_sim: int) -> Dict:
    var_grid = create_var_grid(n_sim)
    split = jnp.split(jnp.arange(n_sim*2*len(var_grid)), len(var_grid))
    key_ints = [jnp.split(subsplit, n_sim) for subsplit in split]
    sim_results = []

    for j, config in enumerate(var_grid):
        parallel_result = Parallel(n_jobs=-2, prefer="threads")(delayed(do_one)(config["response_dist"],
                                                                                config["n_obs"],
                                                                                key_ints[j][i]) for i in range(n_sim))
        store_result, store_coef = zip(*parallel_result)
        results = process_results(store_result, store_coef)
        sim_results.append(results)

    return sim_results

# Create the grid of specifications for the simulation
def create_var_grid(n_sim: int) -> Dict:
    var_dict = {"n_sim": [n_sim],
                "response_dist": ["normal", "bernoulli"],
                "n_obs": [500, 1000, 5000, 10000],
    }

    # Create a list of keys and values from the var_dict
    keys, values = zip(*var_dict.items())

    # Generate all possible combinations of values
    var_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return var_grid

# One body call of the simulation
def do_one(response_dist, n_obs, key_ints) -> Tuple[dict, dict]:
    if response_dist == "normal":
        data_dict = sim_data.normal_quadratic_const(n_obs, key_ints[0])
    elif response_dist == "bernoulli":
        data_dict = sim_data.bernoulli_linear(n_obs, key_ints[0])
    
    model_obj = model_set_up(data_dict["data"], response_dist)
    q = do_inference(model_obj, n_obs, key_ints[1])

    result = return_target(q, response_dist)

    return result, data_dict["coef"]

# Post process the results to obtain Arrays with parameter estimates 
def process_results(store_result: Tuple, store_coef: Tuple) -> Tuple[Dict, Dict]:
    
    results = {}
    coefs = {}

    for data in store_result:
        for key, value in data.items():
            if key in results.keys():
                results[key].append(value)
            else:
                results[key] = [value]

    for data in store_coef:
        for key, value in data.items():
            if key in coefs.keys():
                coefs[key].append(value)
            else:
                coefs[key] = [value]   
    
    for key, value in results.items():
        results[key] = jnp.vstack(value)

    for key, value in coefs.items():
        results[key + "_true"] = jnp.vstack(value)

    return results
 
# Define the model 
def model_set_up(data: Dict, response_dist: Any) -> tiger.ModelGraph:
    if response_dist == "normal":
        # Set up design matrix 
        X = tiger.Obs(name="X_loc")
        X.fixed(data = np.column_stack((data["x"], data["x"]**2)))

        # Set up hyperparameters
        beta_loc = tiger.Hyper(0.0, name="beta_loc")
        beta_scale = tiger.Hyper(100.0, name="beta_scale")

        # Set up parameters
        beta_dist = tiger.Dist(tfjd.Normal, loc=beta_loc, scale=beta_scale)
        beta = tiger.Param(value=np.array([0.0, 0.0, 0.0]), distribution=beta_dist, name="beta")

        # Set up hyperparameters for the scale
        sigma_a = tiger.Hyper(0.01, name="sigma_a")
        sigma_b = tiger.Hyper(0.01, name="sigma_b")

        sigma_dist = tiger.Dist(tfjd.InverseGamma, concentration=sigma_a, scale=sigma_b)

        # Use paramter param_space="positive" to transform sigma into unconstrained space  
        sigma = tiger.Param(value=10.0, distribution=sigma_dist, param_space="positive", name="sigma")

        # Set up the linear predictor
        lpred = tiger.Lpred(obs=X, beta=beta)

        # Set up response distribution
        response_dist = tiger.Dist(tfjd.Normal, loc=lpred, scale=sigma)
        m = tiger.Model(response=data["y"], distribution=response_dist)

        # Set up the DAG 
        graph = tiger.ModelGraph(model=m)
        graph.build_graph()
    
    elif response_dist == "bernoulli":
        # set up design matrix 
        X = tiger.Obs(name = "X")
        X.fixed(data = data["x"])

        # set up hyperparameters
        beta_loc = tiger.Hyper(0, name="beta_loc")
        beta_scale = tiger.Hyper(100, name="beta_scale")

        # set up parameters
        beta_dist = tiger.Dist(tfjd.Normal, loc=beta_loc, scale=beta_scale)
        beta = tiger.Param(value=np.array([0.0, 0.0]), distribution=beta_dist, name="beta")

        # set up the linear predictor
        lpred = tiger.Lpred(obs=X, beta=beta)

        # set up response distribution
        response_dist = tiger.Dist(tfjd.Bernoulli, logits=lpred)
        m = tiger.Model(response=data["y"], distribution=response_dist)
        
        # Set up the DAG 
        graph = tiger.ModelGraph(model=m)
        graph.build_graph()  

    return graph

# Run the inference algorithm 
def do_inference(graph, n_obs, key_int):
    q = bbvi.Bbvi(graph=graph)

    if n_obs <= 1000:
        q.run_bbvi(step_size=0.01,
                   threshold=1e-2,
                   key_int=key_int,
                   batch_size=128,
                   num_var_samples=64,
                   chunk_size=50,
                   epochs=250)
    elif n_obs > 1000:
        q.run_bbvi(step_size=0.001,
                   threshold=1e-2,
                   key_int=key_int,
                   batch_size=128,
                   num_var_samples=64,
                   chunk_size=50,
                   epochs=250)    
    
    return q

def return_target(q, response_dist):
    if response_dist == "normal":
        target = {"loc": q.return_loc_params["beta"]["loc"],
                  "scale": jnp.exp(q.trans_var_params["sigma"]["loc"] + q.trans_var_params["sigma"]["cov"]**2/2)
                 }
    elif response_dist == "bernoulli":
        target = {"logits": q.return_loc_params["beta"]["loc"]}
    
    return target