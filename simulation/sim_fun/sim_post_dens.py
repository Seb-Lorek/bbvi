"""
Kernel density comparison of BBVI and MCMC. 
"""

from . import sim_data

import sys
import os 
import time 
import itertools
import logging 
from joblib import Parallel, delayed

from typing import (
    Union,
    Tuple,
    List,
    Dict,
    Any
)

import jax 
import jax.numpy as jnp

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import liesel.model as lsl
import liesel.goose as gs
from liesel.distributions.mvn_degen import MultivariateNormalDegenerate as LieselMultivariateNormalDegenerate
from liesel.goose.types import Array

from .liesel_helpers import (
    VarianceIG, 
    SplineCoef,
    PSpline, 
    tau2_gibbs_kernel
) 

# Use distributions from tensorflow probability
import tensorflow_probability.substrates.jax.distributions as tfjd
import tensorflow_probability.substrates.jax.bijectors as tfjb

# Set path such that interpreter finds tigerpy
# Assuming we execute the file from the top project directory 
sys.path.append(os.getcwd())

import tigerpy.model as tiger
import tigerpy.bbvi as bbvi
from tigerpy.distributions import MultivariateNormalDegenerate as TigerpyMultivariateNormalDegenerate

# Set the required logging level for liesel (use ERROR to still display error messages, but no warnings)
logger = logging.getLogger("liesel")
logger.setLevel(logging.ERROR)  # Set the level to ERROR for liesel logger

# Create a FileHandler to output logs to the file
log_file = 'simulation/results/sim2_liesel.log'
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)  # Set the level to INFO for the log file

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the FileHandler to the liesel logger
logger.addHandler(file_handler)

# Function to execute the simulation study 
def sim_fun(n_sim: int, n_obs: int, key: jax.Array) -> Tuple[dict, dict]:
    # Create the grid of parameters of the simulation study
    var_grid = create_var_grid(n_obs)

    key, subkey = jax.random.split(key)
    # Create key for data generation 
    data_dict = sim_data.normal_complex_const(n_obs=n_obs, 
                                              key=subkey)
    
    # Set the path to store store the figures in simulation/plots/post_density
    current_path = os.getcwd()
    folder_path = os.path.join(current_path, "simulation/plots/post_density")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created successfully.")
    else:
        print(f"Directory '{folder_path}' already exists.")
    filename = "plot_sim_dat.pdf"
    full_filepath = os.path.join(folder_path, 
                                 filename)
    sim_data.plot_sim_data(data_dict["data"], dist="normal", savepath=full_filepath)

    plot_results = []
    sim_results = {}
    for config in var_grid:
        key, subkey = jax.random.split(key)

        post_results = run_model(n_sim, data_dict, config["library"], subkey)
        plot_results.append(post_results)
        post_samples = process_results(post_results, config["library"])
        sim_results[config["library"]] = post_samples

    # Close the logging FileHandler
    file_handler.close()
    logger.removeHandler(file_handler)

    filename = "plot_post_mean.pdf"
    full_filepath = os.path.join(folder_path, 
                                 filename)
    plot_fns(plot_results, data_dict["data"], full_filepath)

    return sim_results, var_grid

# Create the grid of specifications for the simulation
def create_var_grid(n_obs: int) -> Dict:
    var_dict = {"library": ["tigerpy", "liesel"],
                "n_obs": [n_obs],
    }

    # Create a list of keys and values from the var_dict
    keys, values = zip(*var_dict.items())

    # Generate all possible combinations of values
    var_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return var_grid

def run_model(n_sim: int, data_dict: Dict, library: str, key: jax.Array) -> List:
    key, *subkeys = jax.random.split(key, 3)

    model_obj = model_set_up(data_dict["data"], library, subkeys[0])
    post_results = do_inference(n_sim, model_obj, library, subkeys[1])

    return post_results

def process_results(post_results: Any, library: str) -> jax.Array:
    if library == "tigerpy":
        seq = []
        for q in post_results:
            post_sample = q.get_posterior_samples(sample_shape=1000)
            seq.append(post_sample)
            
        post_samples = {}
        for kw in seq[0].keys():
            post_samples[kw] = None
    
        for kw in post_samples.keys():
            stacked_array = jnp.stack([d[kw] for d in seq], axis=0)
            post_samples[kw] = stacked_array
    elif library == "liesel":
        post_samples = post_results.get_posterior_samples()
        post_samples["gamma"] = post_samples.pop("smooth_1_coef")

    return post_samples

# Define the tiger model
def model_set_up(data: Dict, library: str, key: jax.Array) -> Union[tiger.ModelGraph, gs.EngineBuilder]:
    # In either case we use the model building library of tigerpy to construct the 
    # design matrices 
    X = tiger.Obs(name="X", intercept=True)
    X.smooth(data=data["x"])
    X.center()
    
    if library == "tigerpy":
        beta_loc = tiger.Hyper(0.0, name="beta_loc")
        beta_scale = tiger.Hyper(100.0, name="beta_scale")
        beta_dist = tiger.Dist(tfjd.Normal, loc=beta_loc, scale=beta_scale)
        beta = tiger.Param(value=jnp.array([0.0]), distribution=beta_dist, name="beta")

        tau2_a = tiger.Hyper(1.0, name="tau2_a")
        tau2_b = tiger.Hyper(0.00005, name="tau2_b")
        tau2_dist = tiger.Dist(tfjd.InverseGamma, concentration=tau2_a, scale=tau2_b)
        tau2 = tiger.Param(value=jnp.array([1.0]), distribution=tau2_dist, param_space="positive", name="tau2")

        gamma_loc = tiger.Hyper(jnp.zeros(X.smooth_dim_cent[0]), name="gamma_loc")
        gamma_pen = tiger.Hyper(X.smooth_pen_mat_cent[0], name="gamma_pen")
        gamma_dist = tiger.Dist(TigerpyMultivariateNormalDegenerate, loc=gamma_loc, var=tau2, pen=gamma_pen)
        gamma = tiger.Param(value=jnp.zeros(X.smooth_dim_cent[0]), distribution=gamma_dist, name="gamma")

        lpred = tiger.Lpred(obs=X, beta=beta, gamma=gamma)

        sigma_a = tiger.Hyper(0.01, name="sigma_a")
        sigma_b = tiger.Hyper(0.01, name="sigma_b")
        sigma_dist = tiger.Dist(tfjd.InverseGamma, concentration=sigma_a, scale=sigma_b)
        sigma = tiger.Param(value=10.0, distribution=sigma_dist, param_space="positive", name="sigma")

        response_dist = tiger.Dist(tfjd.Normal, loc=lpred, scale=sigma)
        m = tiger.Model(response=data["y"].to_numpy(), distribution=response_dist)

        final_obj = tiger.ModelGraph(model=m)
        final_obj.build_graph()
    elif library == "liesel":
        beta_loc = lsl.Var(0.0, name="beta_loc")
        beta_scale = lsl.Var(100.0, name="beta_scale")
        beta_dist = lsl.Dist(tfjd.Normal, loc=beta_loc, scale=beta_scale)
        beta = lsl.param(value=jnp.array([0.0]), distribution=beta_dist, name="beta")

        tau2_group = VarianceIG(name="tau2", a=1.0, b=0.00005)
        penalty = X.smooth_pen_mat_cent[0]
        smooth_group_1 = PSpline(name="smooth_1", basis_matrix=X.design_mat_cent[1], penalty=penalty, tau2_group=tau2_group)

        sigma_a = lsl.Var(0.01, name="sigma_a")
        sigma_b = lsl.Var(0.01, name="sigma_b")
        sigma_dist = lsl.Dist(tfjd.InverseGamma, concentration=sigma_a, scale=sigma_b)
        sigma = lsl.param(value=10.0, distribution=sigma_dist, name="sigma")

        Z = lsl.obs(X.fixed_data, name="Z")

        lpred_loc_fn = lambda z, beta, smooth_1: jnp.dot(z, beta) + smooth_1
        lpred_loc_calc = lsl.Calc(lpred_loc_fn, z=Z, beta=beta, smooth_1=smooth_group_1["smooth"])

        lpred_loc = lsl.Var(lpred_loc_calc, name="lpred_loc")

        response_dist = lsl.Dist(tfjd.Normal, loc=lpred_loc, scale=sigma)
        response = lsl.Var(data["y"].to_numpy(), distribution=response_dist, name="response")

        gb = lsl.GraphBuilder().add(response)
        gb.transform(sigma, tfjb.Exp)
        model = gb.build_model()

        builder = gs.EngineBuilder(seed=key, num_chains=4)

        builder.set_model(gs.LieselInterface(model))
        builder.set_initial_values(model.state)

        builder.add_kernel(tau2_gibbs_kernel(smooth_group_1))
        builder.add_kernel(gs.NUTSKernel(["smooth_1_coef"]))
        builder.add_kernel(gs.NUTSKernel(["beta"]))
        builder.add_kernel(gs.NUTSKernel(["sigma_transformed"]))

        builder.set_duration(warmup_duration=1000, posterior_duration=1000)

        builder.positions_included = ["sigma"]

        final_obj = builder.build()
        final_obj._show_progress = False
    return final_obj     

def do_inference(n_sim: int, model_obj: Any, library: str, key: jax.Array) -> Any:
    if library == "tigerpy":
        subkeys_sim = [jax.random.fold_in(key, i) for i in range(n_sim)]
        results = Parallel(n_jobs=-2)(delayed(run_tiger)(model_obj,
                                                         subkeys_sim[i]) for i in range(n_sim))   
    elif library == "liesel":
        model_obj.sample_all_epochs()
        results = model_obj.get_results()

    return results

def run_tiger(model_obj: Any, key: jax.Array) -> Any:
    q = bbvi.Bbvi(graph=model_obj,
                  pre_train=False,
                  jitter_init=True,
                  verbose=False)
    q.run_bbvi(key=key,
               learning_rate=0.01,
               grad_clip=1,
               threshold=1e-2,
               batch_size=256,
               train_share=0.8,
               num_var_samples=64,
               chunk_size=50,
               epochs=500)
    return q

def plot_fns(posteriors: List, data: pd.DataFrame, savepath: str) -> None:

    q = posteriors[0][0]
    p = posteriors[1]

    pred_data_tiger = get_pred_data_tiger(q, data)
    pred_data_liesel = get_pred_data_liesel(p, q, data)
    sort_data = data.sort_values("x")

    # plot the data
    sns.set_theme(style="whitegrid")

    fig = plt.figure(figsize=(8,6))

    plt.scatter(x=data["x"], y=data["y"], s=5)
    plt.plot(sort_data["x"], sort_data["loc"], label="DGP", color=sns.color_palette()[1])
    plt.plot(sort_data["x"], sort_data["loc"] + 1.96*sort_data["scale"], linewidth=0.7, color=sns.color_palette()[1], linestyle='dashed')
    plt.plot(sort_data["x"], sort_data["loc"] - 1.96*sort_data["scale"], linewidth=0.7, color=sns.color_palette()[1], linestyle='dashed')
    plt.plot(pred_data_tiger["x"], pred_data_tiger["y"], color=sns.color_palette()[2], label = "BBVI")
    plt.plot(pred_data_liesel["x"], pred_data_liesel["y"], color=sns.color_palette()[3], label = "MCMC")
    plt.legend()

    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()

def get_pred_data_tiger(q: bbvi.Bbvi, data: pd.DataFrame) -> pd.DataFrame:
    beta = q.trans_var_params["beta"]["loc"]
    gamma = q.trans_var_params["gamma"]["loc"]
    loc_param = jnp.concatenate((beta, gamma))
    y_opt = q.data["X"] @ loc_param
    df = pd.DataFrame({"x": data["x"], "y":y_opt})
    sort_df = df.sort_values("x")

    return sort_df

def get_pred_data_liesel(posterior: gs.SamplingResults, q: bbvi.Bbvi, data: pd.DataFrame) -> pd.DataFrame:
    p = gs.Summary(posterior).quantities
    beta = p["mean"]["beta"]
    gamma = p["mean"]["smooth_1_coef"]
    loc_param = jnp.concatenate((beta, gamma))
    y_opt = q.data["X"] @ loc_param
    df = pd.DataFrame({"x": data["x"], "y":y_opt})
    sort_df = df.sort_values("x")

    return sort_df
