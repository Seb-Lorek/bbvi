"""
Trace plots for the ELBO.
"""

# Dependencies 
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time 
import sys

# Set path such that interpreter finds tigerpy
# Assuming we execute the file from the top project directory 
sys.path.append(os.getcwd())

import tigerpy.model as tiger
import tigerpy.bbvi as bbvi

# Use distributions from tensorflow probability
import tensorflow_probability.substrates.jax.distributions as tfjd

# Set the random seed for numpy 
rng = np.random.default_rng(42)

# Create a jax.array that stores all keys
keys = jnp.arange(5)

# Sample size and true parameters
n = 1000
true_beta = np.array([1.0, 2.0])
true_sigma = 1.0

# Data-generating process
x0 = rng.uniform(size=n)
X_mat = np.column_stack([np.ones(n), x0])
eps = rng.normal(scale=true_sigma, size=n)
response_vec = X_mat @ true_beta + eps

"""
Using different initializations.
"""

# Record time 
start_time = time.time()

# Initial model initialization
# Set up design matrix 
X = tiger.Obs(name="X_loc")
X.fixed(data = x0)

# Set up hyperparameters
beta_loc = tiger.Hyper(0.0, 
                       name="beta_loc")
beta_scale = tiger.Hyper(100.0, 
                         name="beta_scale")

# Set up parameters
beta_dist = tiger.Dist(tfjd.Normal, 
                       loc=beta_loc, 
                       scale=beta_scale)
beta = tiger.Param(value=np.array([0.0, 0.0]), 
                   distribution=beta_dist, 
                   name="beta")

# Set up hyperparameters for the scale
sigma_a = tiger.Hyper(0.01, name="a")
sigma_b = tiger.Hyper(0.01, name="b")

sigma_dist = tiger.Dist(tfjd.InverseGamma, 
                        concentration=sigma_a, 
                        scale=sigma_b)

# Use paramter param_space="positive" to transform sigma into unconstrained space  
sigma = tiger.Param(value=10.0, 
                    distribution=sigma_dist, 
                    param_space="positive", 
                    name="sigma")

# Set up the linear predictor
lpred = tiger.Lpred(obs=X,
                    beta=beta)

# Set up response distribution
response_dist = tiger.Dist(tfjd.Normal,
                           loc=lpred,
                           scale=sigma)
m1 = tiger.Model(response=response_vec, 
                distribution=response_dist)

graph1 = tiger.ModelGraph(model=m1)
graph1.build_graph()

# For the second model use a different mean init for beta and sigma
# Set up hyperparameters
beta_loc = tiger.Hyper(0.0, 
                       name="beta_loc")
beta_scale = tiger.Hyper(100.0, 
                         name="beta_scale")

# Set up parameters
beta_dist = tiger.Dist(tfjd.Normal, 
                       loc=beta_loc, 
                       scale=beta_scale)
# Change beta mean init to 0.5
beta = tiger.Param(value=np.array([0.5, 0.5]), 
                   distribution=beta_dist, 
                   name="beta")

# Set up hyperparameters for the scale
sigma_a = tiger.Hyper(0.01, name="a")
sigma_b = tiger.Hyper(0.01, name="b")

sigma_dist = tiger.Dist(tfjd.InverseGamma, 
                        concentration=sigma_a, 
                        scale=sigma_b)

# Use paramter param_space="positive" to transform sigma into unconstrained space 
# Change sigma mean init to log(5.0)
sigma = tiger.Param(value=5.0, 
                    distribution=sigma_dist, 
                    param_space="positive", 
                    name="sigma")

# Set up the linear predictor
lpred = tiger.Lpred(obs=X,
                    beta=beta)

# Set up response distribution
response_dist = tiger.Dist(tfjd.Normal,
                           loc=lpred,
                           scale=sigma)
m2 = tiger.Model(response=response_vec, 
                distribution=response_dist)

graph2 = tiger.ModelGraph(model=m2)
graph2.build_graph()

# Last model with different var and mean init
# Set up hyperparameters
beta_loc = tiger.Hyper(0.0,
                       name="beta_loc")
beta_scale = tiger.Hyper(100.0,
                         name="beta_scale")

# Set up parameters
beta_dist = tiger.Dist(tfjd.Normal, 
                       loc=beta_loc, 
                       scale=beta_scale)
# Set beta mean again to jnp.zeros
beta = tiger.Param(value=np.array([0.0, 0.0]), 
                   distribution=beta_dist, 
                   name="beta")

# Set up hyperparameters for the scale
sigma_a = tiger.Hyper(0.01, name="a")
sigma_b = tiger.Hyper(0.01, name="b")

sigma_dist = tiger.Dist(tfjd.InverseGamma, 
                        concentration=sigma_a, 
                        scale=sigma_b)

# Use paramter param_space="positive" to transform sigma into unconstrained space  
# Change sigma mean to log(20.0)
sigma = tiger.Param(value=20.0, 
                    distribution=sigma_dist, 
                    param_space="positive", 
                    name="sigma")

# Set up the linear predictor
lpred = tiger.Lpred(obs=X,
                    beta=beta)

# Set up response distribution
response_dist = tiger.Dist(tfjd.Normal,
                           loc=lpred,
                           scale=sigma)
m3 = tiger.Model(response=response_vec, 
                distribution=response_dist)

graph3 = tiger.ModelGraph(model=m3)
graph3.build_graph()

# Run the inference
q1 = bbvi.Bbvi(graph=graph1)
q1.run_bbvi(step_size=0.01,
           threshold=1e-2,
           key_int=keys[0],
           batch_size=128,
           num_var_samples=64,
           chunk_size=50,
           epochs=250)

q2 = bbvi.Bbvi(graph=graph2)
q2.run_bbvi(step_size=0.01,
           threshold=1e-2,
           key_int=keys[1],
           batch_size=128,
           num_var_samples=64,
           chunk_size=50,
           epochs=250)

q3 = bbvi.Bbvi(graph=graph3)
q3.run_bbvi(step_size=0.01,
           threshold=1e-2,
           key_int=keys[2],
           batch_size=128,
           num_var_samples=64,
           chunk_size=50,
           epochs=250)

end_time = time.time()

time_elapsed = end_time - start_time 

print(f"Time elapsed Figure 2:{time_elapsed} seconds")

# set the seaborn theme
sns.set_theme(style="whitegrid")

# Create the plot
fig, ax = plt.subplots(figsize=(8,6))

ax.plot(q1.elbo_hist["epoch"], 
        q1.elbo_hist["elbo"], 
        alpha = 0.8,
        linewidth=1,
        label="run 1")
ax.plot(q2.elbo_hist["epoch"], 
        q2.elbo_hist["elbo"], 
        alpha = 0.8, 
        linewidth=1, 
        label="run 2")
ax.plot(q3.elbo_hist["epoch"], 
        q3.elbo_hist["elbo"], 
        alpha = 0.8, 
        linewidth=1,
        label="run 3")
ax.legend()
plt.title("Convergence of ELBO")
plt.xlabel("Epoch")
plt.ylabel("ELBO")

current_directory = os.getcwd()
path = "thesis/assets/plots"

# Create the full path to the folder
folder_path = os.path.join(current_directory, 
                           path)

filename = "plot2.pdf"

# Create the full path to the file
full_filepath = os.path.join(folder_path, 
                             filename)

# safe figure to assets/plots
fig.savefig(full_filepath)

"""
Using different seeds.
"""

# Record time 
start_time = time.time()

q4 = q1

q5 = bbvi.Bbvi(graph=graph1)
q5.run_bbvi(step_size=0.01,
           threshold=1e-2,
           key_int=keys[3],
           batch_size=128,
           num_var_samples=64,
           chunk_size=50,
           epochs=250)

q6 = bbvi.Bbvi(graph=graph1)
q6.run_bbvi(step_size=0.01,
           threshold=1e-2,
           key_int=keys[4],
           batch_size=128,
           num_var_samples=64,
           chunk_size=50,
           epochs=250)

end_time = time.time()

time_elapsed = end_time - start_time 

print(f"Time elapsed Figure 3:{time_elapsed} seconds")

# Create the plot
fig, ax = plt.subplots(figsize=(8,6))

ax.plot(q4.elbo_hist["epoch"], 
        q4.elbo_hist["elbo"], 
        alpha = 0.8,
        linewidth=1,
        label="run 1")
ax.plot(q5.elbo_hist["epoch"], 
        q5.elbo_hist["elbo"], 
        alpha = 0.8, 
        linewidth=1, 
        label="run 2")
ax.plot(q6.elbo_hist["epoch"], 
        q6.elbo_hist["elbo"], 
        alpha = 0.8, 
        linewidth=1,
        label="run 3")
ax.legend()
plt.title("Convergence of ELBO")
plt.xlabel("Epoch")
plt.ylabel("ELBO")

filename = "plot3.pdf"

# Create the full path to the file
full_filepath = os.path.join(folder_path, 
                             filename)

# safe figure to assets/plots
fig.savefig(full_filepath)