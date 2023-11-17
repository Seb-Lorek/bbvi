"""
Trace plots for the ELBO.
"""

# Dependencies 
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
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

# Create a jax key
# seed generated via secrets.randbits(32) 
key = jax.random.PRNGKey(2829805258)

# Sample size and true parameters
n = 1000
true_beta = jnp.array([1.0, 2.0])
true_sigma = 1.0

key, *subkeys = jax.random.split(key, 3)

# Data-generating process
x0 = jax.random.uniform(subkeys[0], (n,))
X_mat = jnp.column_stack([jnp.ones(n), x0])
eps = tfjd.Normal(loc = 0.0, scale=true_sigma).sample((n,), subkeys[1])
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
beta = tiger.Param(value=jnp.array([0.0, 0.0]), 
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
beta = tiger.Param(value=jnp.array([0.5, 0.5]), 
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
beta = tiger.Param(value=jnp.array([0.0, 0.0]), 
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

# Split the key 
key, subkey = jax.random.split(key)

# Run the inference
q1 = bbvi.Bbvi(graph=graph1, 
               jitter_init=False, 
               verbose=False)
q1.run_bbvi(key=subkey,
            learning_rate=0.01,
            threshold=1e-2,
            batch_size=128,
            num_var_samples=64,
            chunk_size=50,
            epochs=100)

q2 = bbvi.Bbvi(graph=graph2, 
               jitter_init=False, 
               verbose=False)
q2.run_bbvi(key=subkey,
            learning_rate=0.01,
            threshold=1e-2,
            batch_size=128,
            num_var_samples=64,
            chunk_size=50,
            epochs=100)

q3 = bbvi.Bbvi(graph=graph3, 
               jitter_init=False, 
               verbose=False)
q3.run_bbvi(key=subkey,
            learning_rate=0.01,
            threshold=1e-2,
            batch_size=128,
            num_var_samples=64,
            chunk_size=50,
            epochs=100)

end_time = time.time()

time_elapsed = end_time - start_time 

print(f"Time elapsed Figure 2:{time_elapsed:.2f} seconds")

# set the seaborn theme
sns.set_theme(style="whitegrid")

# Create the plot
fig, ax = plt.subplots(figsize=(8,6))

ax.plot(q1.elbo_hist["epoch"], 
        q1.elbo_hist["elbo_epoch"], 
        alpha = 0.6,
        linewidth=1,
        color=sns.color_palette("colorblind")[0],
        label="run 1")
ax.plot(q2.elbo_hist["epoch"], 
        q2.elbo_hist["elbo_epoch"], 
        alpha = 0.6, 
        linewidth=1, 
        color=sns.color_palette("colorblind")[1],
        label="run 2")
ax.plot(q3.elbo_hist["epoch"], 
        q3.elbo_hist["elbo_epoch"], 
        alpha = 0.6, 
        linewidth=1,
        color=sns.color_palette("colorblind")[2],
        label="run 3")
ax.legend(loc="lower right")
plt.title("Different init.")
plt.xlabel("Epoch")
plt.ylabel("ELBO")
a = plt.axes((.525, .525, .35, .25))
plt.plot(q1.elbo_hist["epoch"], 
        q1.elbo_hist["elbo_epoch"], 
        alpha = 0.6,
        linewidth=1,
        color=sns.color_palette("colorblind")[0],
        label="run 1")
plt.plot(q2.elbo_hist["epoch"], 
        q2.elbo_hist["elbo_epoch"], 
        alpha = 0.6, 
        linewidth=1, 
        color=sns.color_palette("colorblind")[1],
        label="run 2")
plt.plot(q3.elbo_hist["epoch"], 
        q3.elbo_hist["elbo_epoch"], 
        alpha = 0.6, 
        linewidth=1,
        color=sns.color_palette("colorblind")[2],
        label="run 3")
plt.ylim(-310, -305)

current_directory = os.getcwd()
path1 = "thesis/assets/plots"
path2 = "simulation/plots/trace_elbo"

# Create the full path to the folder
folder_path1 = os.path.join(current_directory, 
                            path1)
folder_path2 = os.path.join(current_directory, 
                            path2)

filename = "plot2.pdf"

# Create the full path to the file
full_filepath1 = os.path.join(folder_path1, 
                             filename)
full_filepath2 = os.path.join(folder_path2, 
                             filename)

# safe figure to assets/plots
fig.savefig(full_filepath1)
fig.savefig(full_filepath2)

"""
Using different seeds.
"""

# Record time 
start_time = time.time()

q4 = q1

# Split the key 
key, *subkeys = jax.random.split(key, 3)

q5 = bbvi.Bbvi(graph=graph1, 
               jitter_init=False, 
               verbose=False)
q5.run_bbvi(key=subkeys[0],
            learning_rate=0.01,
            threshold=1e-2,
            batch_size=128,
            num_var_samples=64,
            chunk_size=50,
            epochs=100)

q6 = bbvi.Bbvi(graph=graph1, 
               jitter_init=False, 
               verbose=False)
q6.run_bbvi(key=subkeys[1],
            learning_rate=0.01,
            threshold=1e-2,
            batch_size=128,
            num_var_samples=64,
            chunk_size=50,
            epochs=100)
 
end_time = time.time()

time_elapsed = end_time - start_time 

print(f"Time elapsed Figure 3:{time_elapsed:.2f} seconds")

# Create the plot
fig, ax = plt.subplots(figsize=(8,6))

ax.plot(q4.elbo_hist["epoch"], 
        q4.elbo_hist["elbo_epoch"], 
        alpha = 0.8,
        linewidth=1,
        color=sns.color_palette("colorblind")[0],
        label="run 1")
ax.plot(q5.elbo_hist["epoch"], 
        q5.elbo_hist["elbo_epoch"], 
        alpha = 0.8, 
        linewidth=1, 
        color=sns.color_palette("colorblind")[1],
        label="run 2")
ax.plot(q6.elbo_hist["epoch"], 
        q6.elbo_hist["elbo_epoch"], 
        alpha = 0.8, 
        linewidth=1,
        color=sns.color_palette("colorblind")[2],
        label="run 2")
ax.legend(loc="lower right")
plt.title("Different seeds")
plt.xlabel("Epoch")
plt.ylabel("ELBO")

filename = "plot3.pdf"

# Create the full path to the file
full_filepath1 = os.path.join(folder_path1, 
                             filename)
full_filepath2 = os.path.join(folder_path2, 
                             filename)

# safe figure to assets/plots
fig.savefig(full_filepath1)
fig.savefig(full_filepath2)


"""
Change the variational samples size 1, 32, 64, with batch VI.
"""

# Use subkey from Figure 2

# Record time 
start_time = time.time()

q7 = bbvi.Bbvi(graph=graph1, 
               jitter_init=False, 
               verbose=False)
q7.run_bbvi(key=subkey,
            learning_rate=0.01,
            threshold=1e-2,
            batch_size=round(n*0.8),
            num_var_samples=1,
            chunk_size=50,
            epochs=500)

q8 = bbvi.Bbvi(graph=graph1, 
               jitter_init=False, 
               verbose=False)
q8.run_bbvi(key=subkey,
            learning_rate=0.01,
            threshold=1e-2,
            batch_size=round(n*0.8),
            num_var_samples=32,
            chunk_size=50,
            epochs=500)

q9 = bbvi.Bbvi(graph=graph1, 
               jitter_init=False, 
               verbose=False)
q9.run_bbvi(key=subkey,
            learning_rate=0.01,
            threshold=1e-2,
            batch_size=round(n*0.8),
            num_var_samples=64,
            chunk_size=50,
            epochs=500)

end_time = time.time()

time_elapsed = end_time - start_time 

print(f"Time elapsed Figure 4:{time_elapsed:.2f} seconds")

# Create the plot
fig, ax = plt.subplots(figsize=(8,6))

ax.plot(q7.elbo_hist["epoch"], 
        q7.elbo_hist["elbo_epoch"], 
        alpha = 0.8,
        linewidth=1,
        color=sns.color_palette("colorblind")[0],
        label="samples=1")
ax.plot(q8.elbo_hist["epoch"], 
        q8.elbo_hist["elbo_epoch"], 
        alpha = 0.8, 
        linewidth=1, 
        color=sns.color_palette("colorblind")[1],
        label="samples=32")
ax.plot(q9.elbo_hist["epoch"], 
        q9.elbo_hist["elbo_epoch"], 
        alpha = 0.8, 
        linewidth=1,
        color=sns.color_palette("colorblind")[2],
        label="samples=64")

ax.legend(loc="lower right")
plt.ylim(-500)
plt.title("Different var. sample size")
plt.xlabel("Epoch")
plt.ylabel("ELBO")

filename = "plot4.pdf"

# Create the full path to the file
full_filepath1 = os.path.join(folder_path1, 
                              filename)
full_filepath2 = os.path.join(folder_path2, 
                              filename)

# safe figure to assets/plots
fig.savefig(full_filepath1)
fig.savefig(full_filepath2)

"""
Change the batch-size 128, 256, 512, with variational samples of size=64.
"""

# Record time 
start_time = time.time()

q10 = bbvi.Bbvi(graph=graph1, 
                jitter_init=False, 
                verbose=False)
q10.run_bbvi(key=subkey,
             learning_rate=0.01,
             threshold=1e-2,
             batch_size=36,
             num_var_samples=64,
             chunk_size=50,
             epochs=500)

q11 = bbvi.Bbvi(graph=graph1, 
                jitter_init=False, 
                verbose=False)
q11.run_bbvi(key=subkey,
             learning_rate=0.01,
             threshold=1e-2,
             batch_size=128,
             num_var_samples=64,
             chunk_size=50,
             epochs=750)

q12 = bbvi.Bbvi(graph=graph1, 
                jitter_init=False, 
                verbose=False)
q12.run_bbvi(key=subkey,
             learning_rate=0.01,
             threshold=1e-2,
             batch_size=512,
             num_var_samples=64,
             chunk_size=50,
             epochs=1000)

end_time = time.time()

time_elapsed = end_time - start_time 

print(f"Time elapsed Figure 5:{time_elapsed:.2f} seconds")

# Create the plot
fig, ax = plt.subplots(figsize=(8,6))

ax.plot(q10.elbo_hist["iteration"][:500], 
        q10.elbo_hist["elbo_full_val"][:500], 
        alpha = 0.8,
        linewidth=1,
        color=sns.color_palette("colorblind")[0],
        label="batch size=32")
ax.plot(q11.elbo_hist["iteration"][:500], 
        q11.elbo_hist["elbo_full_val"][:500], 
        alpha = 0.8, 
        linewidth=1, 
        color=sns.color_palette("colorblind")[1],
        label="batch size=128")
ax.plot(q12.elbo_hist["iteration"][:500], 
        q12.elbo_hist["elbo_full_val"][:500], 
        alpha = 0.8, 
        linewidth=1,
        color=sns.color_palette("colorblind")[2],
        label="batch size=512")
ax.legend(loc="lower right")
plt.ylim(-500)
plt.title("Different batch size")
plt.xlabel("Iteration")
plt.ylabel("ELBO")

filename = "plot5.pdf"

# Create the full path to the file
full_filepath1 = os.path.join(folder_path1, 
                              filename)
full_filepath2 = os.path.join(folder_path2, 
                              filename)

# safe figure to assets/plots
fig.savefig(full_filepath1)
fig.savefig(full_filepath2)