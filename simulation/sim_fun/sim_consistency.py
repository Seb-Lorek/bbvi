"""
Simulation for convergence in mean-square. 
"""

from . import sim_data

import itertools
import sys
import os
import time 
import logging

from typing import (
    Union,
    Tuple,
    List,
    Dict,
    Any
)

from joblib import Parallel, delayed

import numpy as np
import jax
import jax.numpy as jnp

# Use distributions from tensorflow probability
import tensorflow_probability.substrates.jax.distributions as tfjd
import tensorflow_probability.substrates.jax.bijectors as tfjb

# Set path such that interpreter finds tigerpy
# Assuming we execute the file from the top project directory 
sys.path.append(os.getcwd())

import tigerpy.model as tiger
import tigerpy.bbvi as bbvi

import liesel.model as lsl
import liesel.goose as gs

# Set the required logging level for liesel (use ERROR to still display error messages, but no warnings)
logger = logging.getLogger("liesel")
logger.setLevel(logging.ERROR)  # Set the level to ERROR for liesel logger

# Create a FileHandler to output logs to the file
log_file = 'simulation/results/sim1_liesel.log'
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)  # Set the level to INFO for the log file

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the FileHandler to the liesel logger
logger.addHandler(file_handler)

# Function to execute the simulation study 
def sim_fun(n_sim: int, key: jax.random.PRNGKey) -> Dict:
    # Create the grid of parameters of the simulation study
    var_grid = create_var_grid(n_sim)

    # Create an array of seeds 
    subkeys_grid = [jax.random.fold_in(key, i) for i in range(len(var_grid))]
    sim_results = []

    for j, config in enumerate(var_grid):
        subkeys_sim = [jax.random.fold_in(subkeys_grid[j], i) for i in range(n_sim)]
        parallel_result = Parallel(n_jobs=-2)(delayed(do_one)(config["response_dist"],
                                                              config["n_obs"],
                                                              subkeys_sim[i]) for i in range(n_sim))
        
        store_tiger, store_lsl, store_coef = zip(*parallel_result)
        results = process_results(store_tiger, store_lsl, store_coef)
        sim_results.append(results)

    # Close the logging FileHandler
    file_handler.close()
    logger.removeHandler(file_handler)
        
    return sim_results, var_grid

# Create the grid of specifications for the simulation
def create_var_grid(n_sim: int) -> Dict:
    var_dict = {"n_sim": [n_sim],
                "response_dist": ["normal", "bernoulli"],
                "n_obs": [50, 100, 500, 1000, 5000]
                }

    # Create a list of keys and values from the var_dict
    keys, values = zip(*var_dict.items())

    # Generate all possible combinations of values
    var_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return var_grid

# One body call of the simulation
def do_one(response_dist: str, n_obs: int, key: jax.random.PRNGKey) -> Tuple[dict, dict, dict]:
    key, subkey = jax.random.split(key)
    if response_dist == "normal":
        data_dict = sim_data.normal_quadratic_const(n_obs, subkey)
    elif response_dist == "bernoulli":
        data_dict = sim_data.bernoulli_linear(n_obs, subkey)
    
    results = {}
    libs = ["tigerpy", "liesel"]
    for lib in libs:
        key, *subkeys = jax.random.split(key, 3)
        model_obj = model_set_up(lib, data_dict["data"], response_dist, subkeys[0])
        p = do_inference(lib, model_obj, subkeys[1])
        result = return_target(lib, p, response_dist)
        results[lib] = result

    return results["tigerpy"], results["liesel"], data_dict["coef"]

# Post process the results to obtain Arrays with parameter estimates 
def process_results(store_result_tiger: Tuple, 
                    store_result_liesel: Tuple, 
                    store_coef: Tuple) -> Dict:
    libs = ["tigerpy", "liesel"]
    results = {lib: {} for lib in libs}
    coefs = {lib: {} for lib in libs}

    # Unpack tiger 
    for data in store_result_tiger:
        for kw, value in data.items():
            if kw in results["tigerpy"].keys():
                results["tigerpy"][kw].append(value)
            else:
                results["tigerpy"][kw] = [value]

    # Unpack liesel 
    for data in store_result_liesel:
        for kw, value in data.items():
            if kw in results["liesel"].keys():
                results["liesel"][kw].append(value)
            else:
                results["liesel"][kw] = [value]

    for data in store_coef:
        for lib in libs:
            for kw, value in data.items():
                if kw in coefs.keys():
                    coefs[lib][kw].append(value)
                else:
                    coefs[lib][kw] = [value]  

    for lib in libs:
        for kw, value in results[lib].items():
            results[lib][kw] = jnp.vstack(value)
        for kw, value in coefs[lib].items():
            results[lib][kw + "_true"] = jnp.vstack(value)

    return results
 
# Define the model 
def model_set_up(lib: str, data: Dict, response_dist: Any, key: jax.Array) -> tiger.ModelGraph:
    if lib == "tigerpy":
        if response_dist == "normal":
            # Set up design matrix 
            X = tiger.Obs(name="X")
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
            m = tiger.Model(response=data["y"].to_numpy(), distribution=response_dist)

            # Set up the DAG 
            final_obj = tiger.ModelGraph(model=m)
            final_obj.build_graph()

        elif response_dist == "bernoulli":
            # Set up design matrix 
            X = tiger.Obs(name="X")
            X.fixed(data =data["x"].to_numpy())

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
            final_obj = tiger.ModelGraph(model=m)
            final_obj.build_graph()  
    elif lib == "liesel":
        if response_dist == "normal":
            # Set up design matrix 
            X = tiger.Obs(name="X")
            X.fixed(data = np.column_stack((data["x"], data["x"]**2)))

            beta_loc = lsl.Var(0.0, name="beta_loc")
            beta_scale = lsl.Var(100.0, name="beta_scale")
            beta_dist = lsl.Dist(tfjd.Normal, loc=beta_loc, scale=beta_scale)
            beta = lsl.param(value=jnp.array([0.0, 0.0, 0.0]), distribution=beta_dist, name="beta")

            sigma_a = lsl.Var(0.01, name="sigma_a")
            sigma_b = lsl.Var(0.01, name="sigma_b")
            sigma_dist = lsl.Dist(tfjd.InverseGamma, concentration=sigma_a, scale=sigma_b)
            sigma = lsl.param(value=10.0, distribution=sigma_dist, name="sigma")

            Z = lsl.obs(X.fixed_data, name="Z")
            lpred_loc_fn = lambda z, beta: jnp.dot(z, beta)
            lpred_loc_calc = lsl.Calc(lpred_loc_fn, z=Z, beta=beta)
            lpred_loc = lsl.Var(lpred_loc_calc, name="lpred_loc")

            response_dist = lsl.Dist(tfjd.Normal, loc=lpred_loc, scale=sigma)
            response = lsl.Var(data["y"].to_numpy(), distribution=response_dist, name="response")

            gb = lsl.GraphBuilder().add(response)
            gb.transform(sigma, tfjb.Exp)
            model = gb.build_model()

            builder = gs.EngineBuilder(seed=key, num_chains=4)

            builder.set_model(gs.LieselInterface(model))
            builder.set_initial_values(model.state)

            builder.add_kernel(gs.NUTSKernel(["beta"]))
            builder.add_kernel(gs.NUTSKernel(["sigma_transformed"]))

            builder.set_duration(warmup_duration=1000, posterior_duration=1000)

            builder.positions_included = ["sigma"]

            final_obj = builder.build()
            final_obj._show_progress = False
        elif response_dist == "bernoulli":
            # Set up design matrix 
            X = tiger.Obs(name="X")
            X.fixed(data =data["x"].to_numpy())

            beta_loc = lsl.Var(0.0, name="beta_loc")
            beta_scale = lsl.Var(100.0, name="beta_scale")
            beta_dist = lsl.Dist(tfjd.Normal, loc=beta_loc, scale=beta_scale)
            beta = lsl.param(value=jnp.array([0.0, 0.0]), distribution=beta_dist, name="beta")

            Z = lsl.obs(X.fixed_data, name="Z")
            lpred_loc_fn = lambda z, beta: jnp.dot(z, beta)
            lpred_loc_calc = lsl.Calc(lpred_loc_fn, z=Z, beta=beta)
            lpred_loc = lsl.Var(lpred_loc_calc, name="lpred_loc")

            response_dist = lsl.Dist(tfjd.Bernoulli, logits=lpred_loc)
            response = lsl.Var(data["y"].to_numpy(), distribution=response_dist, name="response")

            gb = lsl.GraphBuilder().add(response)
            model = gb.build_model()

            builder = gs.EngineBuilder(seed=key, num_chains=4)

            builder.set_model(gs.LieselInterface(model))
            builder.set_initial_values(model.state)

            builder.add_kernel(gs.NUTSKernel(["beta"]))

            builder.set_duration(warmup_duration=1000, posterior_duration=1000)

            final_obj = builder.build()
            final_obj._show_progress = False

    return final_obj

# Run the inference algorithm 
def do_inference(lib: str, model_obj: Any, key: jax.Array) -> Any:
    if lib == "tigerpy":
        result = bbvi.Bbvi(graph=model_obj,
                           pre_train=False,
                           jitter_init=True,
                           verbose=False)
        result.run_bbvi(key=key,
                        learning_rate=0.01,
                        grad_clip=1,
                        threshold=1e-2,
                        batch_size=36,
                        train_share=0.8,
                        num_var_samples=64,
                        chunk_size=50,
                        epochs=250)
    elif lib == "liesel":
        model_obj.sample_all_epochs()
        p = model_obj.get_results()
        p_moments = gs.Summary(p).quantities
        result = p_moments["mean"]
    return result

def return_target(lib, model_obj, response_dist):
    if lib == "tigerpy":
        if response_dist == "normal":
            target = {"loc": model_obj.return_loc_params["beta"]["loc"],
                      "scale": model_obj.return_loc_params["sigma"]["loc"]}
        elif response_dist == "bernoulli":
            target = {"logits": model_obj.return_loc_params["beta"]["loc"]}
    elif lib == "liesel":
        if response_dist == "normal":
            target = {"loc": model_obj["beta"], 
                      "scale": model_obj["sigma"]}
        elif response_dist == "bernoulli":
            target = {"logits": model_obj["beta"]}
    
    return target