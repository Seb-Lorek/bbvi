"""
Simulation for convergence in mean-square. 
"""

import sim_data

import itertools
import sys
import os
import time 

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfjd

# Set path such that interpreter finds tigerpy
# Assuming we execute the file from the top project directory 
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "..", ".."))

import tigerpy.model as tiger
import tigerpy.bbvi as bbvi

# Function to execute the simulation study 
def sim_fun(n_sim):
    var_grid = create_var_grid(n_sim)
    key_ints = jnp.array_split(jnp.arange(n_sim*2), n_sim)
    sim_results = []
    true_coefs = []

    for config in var_grid:
        store_result = []
        store_coef = [] 
        for i in range(n_sim):
            result, true_coef = do_one(config["response_dist"], config["n_obs"], key_ints[i])
            store_result.append(result)
            store_coef.append(true_coef)
        true_coefs.append(store_coef)
        sim_results.append(store_result)

    return sim_results

# Create the grid of specifications for the simulation
def create_var_grid(n_sim):
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
def do_one(response_dist, n_obs, key_ints):
    if response_dist == "normal":
        data_dict = sim_data.normal_quadratic_const(n_obs, key_ints[0])
    elif response_dist == "bernoulli":
        data_dict = sim_data.bernoulli_linear(n_obs, key_ints[0])
    
    model_obj = model_set_up(data_dict["data"], response_dist)
    q = do_inference(model_obj, n_obs, key_ints[1])

    results = return_target(q, response_dist)

    return results, data_dict["coef"]

# Define the model 
def model_set_up(data, response_dist):
    if response_dist == "normal":
        # Set up design matrix 
        X = tiger.Obs(name="X_loc")
        X.fixed(data = np.column_stack([data["x"], data["x"]**2]))

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

# Test the functions 
test = sim_data.sim_data_normal([-3,3], 
                                500, 
                                sim_data.quadratic_fun,
                                None,
                                coef_loc=jnp.array([3.0, 0.2, -0.5]),
                                coef_scale=jnp.array(1.0),
                                key_int=0)

sim_data.plot_sim_data(test, dist="normal")

# Test the functions 
test = sim_data.sim_data_bernoulli([-3,3], 
                                   500, 
                                   sim_data.linear_fun,
                                   coef_logits=jnp.array([1.0, 2.0]),
                                   key_int=2)

sim_data.plot_sim_data(test, dist="bernoulli")

# test some functions 
test_grid = create_var_grid(250)
print(pd.DataFrame(test_grid))
test1 = do_one(test_grid[3]["response_dist"], test_grid[3]["n_obs"], [0,1])
test2 = do_one(test_grid[7]["response_dist"], test_grid[4]["n_obs"], [2,3])

print(test1, 
      test2)

# Try with small n_sim=10 
start_time = time.time()

test_ten = sim_fun(10)

end_time = time.time()

time_elapsed = end_time - start_time 

print(f"Time elapsed for n_sim=10:{time_elapsed} seconds")

print(test_ten)