"""
Compare run time of BBVI and MCMC.
"""

import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os 
import logging
import time

from typing import (
    Union,
    Tuple,
    List,
    Dict,
    Any
)

# Set path such that interpreter finds tigerpy
# Assuming we execute the file from the top project directory 

sys.path.append(os.getcwd())

import tigerpy.model as tiger
import tigerpy.bbvi as bbvi
from tigerpy.distributions import MultivariateNormalDegenerate as tigerpyMVNDG

import liesel.model as lsl
import liesel.goose as gs
from liesel.distributions import MultivariateNormalDegenerate as lieselMVNDG
from liesel.goose.types import Array

from .liesel_helpers import (
    VarianceIG, 
    SplineCoef,
    PSpline, 
    tau2_gibbs_kernel
) 

# We use distributions and bijectors from tensorflow probability
import tensorflow_probability.substrates.jax.distributions as tfjd
import tensorflow_probability.substrates.jax.bijectors as tfjb

# Set the required logging level for liesel (use ERROR to still display error messages, but no warnings)
logger = logging.getLogger("liesel")
logger.setLevel(logging.ERROR)  # Set the level to ERROR for liesel logger

# Create a FileHandler to output logs to the file
log_file = 'simulation/results/sim3_liesel.log'
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)  # Set the level to INFO for the log file

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the FileHandler to the liesel logger
logger.addHandler(file_handler)

def sim_fun(n_sim: int, key: jax.random.PRNGKey) -> Dict:
    # Load the data
    df = pd.read_csv("data/dbbmi.csv", sep=",")
    tiger_time = jnp.array([])
    liesel_time = jnp.array([])
    for _ in range(n_sim):
        key, *subkeys = jax.random.split(key, 3)

        tiger_run = run_tiger(df, subkeys[0])
        tiger_time = jnp.append(tiger_time, tiger_run)
        liesel_run = run_liesel(df, subkeys[1])
        liesel_time = jnp.append(liesel_time, liesel_run)

    results = {"tigerpy": tiger_time, "liesel": liesel_time}

    return results

def run_tiger(df: pd.DataFrame, key: jax.Array):
    start_time = time.time()
    subkeys = jax.random.split(key, 2)
    model = model_set_up("tigerpy", df, subkeys[0])
    _ = do_inference("tigerpy", model, subkeys[1])
    end_time = time.time()
    time_elapsed = end_time - start_time

    return time_elapsed/60

def run_liesel(df: pd.DataFrame, key: jax.Array):
    start_time = time.time()
    subkeys = jax.random.split(key, 2)
    model = model_set_up("liesel", df, subkeys[0])
    _ = do_inference("liesel", model, subkeys[1])
    end_time = time.time()
    time_elapsed = end_time - start_time 

    return time_elapsed/60

def model_set_up(lib: str, df: pd.DataFrame, key: jax.Array):
    # Define a model with intercept 
    # Set up design matrix for loc
    X = tiger.Obs(name="X", intercept=True)
    X.smooth(data=df["age"].to_numpy(), n_knots=25)
    # If we combine fixed covariates and smooth covariates we need to center the the smooth effects first 
    X.center()
    # Set up design matrix for scale 
    Z = tiger.Obs(name="Z", intercept=True)
    Z.smooth(data=df["age"].to_numpy(), n_knots=25)
    # If we combine fixed covariates and smooth covariates we need to center the the smooth effects first 
    Z.center()
    if lib == "tigerpy":
        # Linear Predictor for the location ----
        # Fixed coefficents 
        # Set up beta_fixed hyperparameters 
        beta_loc = tiger.Hyper(0.0, name="beta_loc")
        beta_scale = tiger.Hyper(100.0, name="beta_scale")

        # Set up parameters
        beta_dist = tiger.Dist(tfjd.Normal, loc=beta_loc, scale=beta_scale)
        beta = tiger.Param(value=df["bmi"].to_numpy().mean(), distribution=beta_dist, name="beta")

        # Smooth coefficients
        # Set up hyperparameters for the beta_smooth_scale  
        smooth1_tau2_a = tiger.Hyper(1.0, name="smooth1_tau2_a")
        smooth1_tau2_b = tiger.Hyper(0.00005, name="smooth1_tau2_b")
        smooth1_tau2_dist = tiger.Dist(tfjd.InverseGamma, concentration=smooth1_tau2_a, scale=smooth1_tau2_b)
        smooth1_tau2 = tiger.Param(value=jnp.array([1.0]), distribution=smooth1_tau2_dist, param_space="positive", name="smooth1_tau2")

        # Set up smooth coefficients with mvn degenerate 
        # Set up hyperparameters
        smooth1_loc = tiger.Hyper(np.zeros(X.smooth_dim_cent[0]), name="smooth1_loc")
        smooth1_pen = tiger.Hyper(X.smooth_pen_mat_cent[0], name="smooth1_pen")
        # Set up parameters
        smooth1_dist = tiger.Dist(tigerpyMVNDG, loc=smooth1_loc, var=smooth1_tau2, pen=smooth1_pen)
        smooth1 = tiger.Param(value=np.zeros(X.smooth_dim_cent[0]), distribution=smooth1_dist, name="smooth1")

        # Set up the linear predictor
        lpred_loc = tiger.Lpred(obs=X, beta_fixed=beta, beta_smooth=smooth1)

        # Linear Predictor for the scale ----
        # Fixed coefficents 
        # Set up beta_fixed hyperparameters 
        gamma_loc = tiger.Hyper(0.0, name="gamma_loc")
        gamma_scale = tiger.Hyper(3.0, name="gamma_scale")

        # Set up parameters
        gamma_dist = tiger.Dist(tfjd.Normal, loc=gamma_loc, scale=gamma_scale)
        gamma = tiger.Param(value=np.log(df["bmi"].to_numpy().std()), distribution=gamma_dist, name="gamma")

        # Smooth coefficients
        # Set up hyperparameters for the beta_smooth_scale  
        smooth2_tau2_a = tiger.Hyper(1.0, name="smooth2_tau2_a")
        smooth2_tau2_b = tiger.Hyper(0.00005, name="smooth2_tau2_b")
        smooth2_tau2_dist = tiger.Dist(tfjd.InverseGamma, concentration=smooth2_tau2_a, scale=smooth2_tau2_b)
        smooth2_tau2 = tiger.Param(value=jnp.array([1.0]), distribution=smooth2_tau2_dist, param_space="positive", name="smooth2_tau2")

        # Set up smooth coefficients with mvn degenerate 
        # Set up hyperparameters
        smooth2_loc = tiger.Hyper(np.zeros(Z.smooth_dim_cent[0]), name="smooth2_loc")
        smooth2_pen = tiger.Hyper(Z.smooth_pen_mat_cent[0], name="smooth2_pen")
        # Set up parameters
        smooth2_dist = tiger.Dist(tigerpyMVNDG, loc=smooth2_loc, var=smooth2_tau2, pen=smooth2_pen)
        smooth2 = tiger.Param(value=np.zeros(Z.smooth_dim_cent[0]), distribution=smooth2_dist, name="smooth2")

        # ----
        # Set up the linear predictor
        lpred_scale = tiger.Lpred(obs=Z, gamma_fixed=gamma, gamma_smooth=smooth2, function=jnp.exp)

        # Set up response distribution
        response_dist = tiger.Dist(tfjd.Normal, loc=lpred_loc, scale=lpred_scale)
        m = tiger.Model(response=df["bmi"].to_numpy(), distribution=response_dist)

        # Set up the DAG 
        final_obj = tiger.ModelGraph(model=m)
        final_obj.build_graph()
    elif lib == "liesel":
        # Set up model in liesel
        # Loc branch 
        # Fixed parameter prior
        beta_loc = lsl.Var(0.0, name="beta_loc")
        beta_scale = lsl.Var(100.0, name="beta_scale")

        # Set up the fixed parameters
        beta_dist = lsl.Dist(tfjd.Normal, loc=beta_loc, scale=beta_scale)
        beta = lsl.param(value=jnp.zeros((1,)), distribution=beta_dist, name="beta")

        # Set up the smooth parameters
        tau2_group_1 = VarianceIG(name="tau2_smooth_1", a=1.0, b=0.00005)

        penalty = X.smooth_pen_mat_cent[0]
        smooth_group_1 = PSpline(name="smooth_1", basis_matrix=X.design_mat_cent[1], penalty=penalty, tau2_group=tau2_group_1)

        X_liesel = lsl.obs(X.fixed_data, name="X")

        loc_fn = lambda x, beta, smooth_1: jnp.dot(x, beta) + smooth_1
        loc_calc = lsl.Calc(loc_fn, x=X_liesel, beta=beta, smooth_1=smooth_group_1["smooth"])

        loc = lsl.Var(loc_calc, name="loc")

        # Scale branch 
        # Fixed parameter prior
        gamma_loc = lsl.Var(0.0, name="gamma_loc")
        gamma_scale = lsl.Var(10.0, name="gamma_scale")

        # Set up the fixed parameters
        gamma_dist = lsl.Dist(tfjd.Normal, loc=gamma_loc, scale=gamma_scale)
        gamma = lsl.param(value=jnp.zeros((1,)), distribution=gamma_dist, name="gamma")

        Z_liesel = lsl.obs(Z.fixed_data, name="Z")

        # Set up the smooth parameters
        tau2_group_2 = VarianceIG(name="tau2_smooth_2", a=1.0, b=0.00005)

        penalty = Z.smooth_pen_mat_cent[0]
        smooth_group_2 = PSpline(name="smooth_2", basis_matrix=Z.design_mat_cent[1], penalty=penalty, tau2_group=tau2_group_2)

        lpred_scale_fn = lambda z, gamma, smooth_2: jnp.dot(z, gamma) + smooth_2
        lpred_scale_calc = lsl.Calc(lpred_scale_fn, z=Z_liesel, gamma=gamma, smooth_2=smooth_group_2["smooth"])
        lpred_scale = lsl.Var(lpred_scale_calc, name="lpred_scale")

        scale_fn = lambda s: jnp.exp(s)
        scale_calc = lsl.Calc(scale_fn, s=lpred_scale)
        scale = lsl.Var(scale_calc, name="scale")

        response_dist = lsl.Dist(tfjd.Normal, loc=loc, scale=scale)
        response = lsl.Var(df["bmi"].to_numpy(), distribution=response_dist, name="response")

        gb = lsl.GraphBuilder().add(response)
        model = gb.build_model()

        builder = gs.EngineBuilder(seed=key, num_chains=4)

        builder.set_model(gs.LieselInterface(model))
        builder.set_initial_values(model.state)

        builder.add_kernel(tau2_gibbs_kernel(smooth_group_1))
        builder.add_kernel(tau2_gibbs_kernel(smooth_group_2))
        builder.add_kernel(gs.NUTSKernel(["beta"]))
        builder.add_kernel(gs.NUTSKernel(["smooth_1_coef"]))
        builder.add_kernel(gs.NUTSKernel(["gamma"]))
        builder.add_kernel(gs.NUTSKernel(["smooth_2_coef"]))

        builder.set_duration(warmup_duration=1000, posterior_duration=1000)

        final_obj = builder.build()
        final_obj._show_progress = False

    return final_obj


def do_inference(lib: str, model_obj: Any, key: jax.Array) -> Any:
    if lib == "tigerpy":
        result = bbvi.Bbvi(graph=model_obj,
                           pre_train=True,
                           verbose=False)
        result.run_bbvi(key=key,
                        learning_rate=0.01,
                        pre_train_learning_rate=0.01,
                        grad_clip=1,
                        threshold=1e-2,
                        pre_train_threshold=1,
                        batch_size=256,
                        pre_train_batch_size=256,
                        train_share=0.8,
                        num_var_samples=64,
                        chunk_size=50,
                        epochs=250)
    elif lib == "liesel":
        model_obj.sample_all_epochs()
        result = model_obj.get_results()

    return result