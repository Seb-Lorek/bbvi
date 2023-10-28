import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
import os

# Set path such that interpreter finds tigerpy
sys.path.append(os.path.join(os.getcwd(), ".."))

import liesel.goose as gs
import liesel.model as lsl
from liesel.distributions.mvn_degen import MultivariateNormalDegenerate as LieselMultivariateNormalDegenerate
from liesel.goose.types import Array

import tigerpy.model as tiger
import tigerpy.bbvi as bbvi
from tigerpy.distributions import MultivariateNormalDegenerate as TigerpyMultivariateNormalDegenerate

# Use distributions from tensorflow probability
import tensorflow_probability.substrates.jax.distributions as tfjd
import tensorflow_probability.substrates.jax.bijectors as tfjb

import simulation.sim_fun.sim_data as sim_data#

data = sim_data.normal_complex_const(1000, key_int=0)

data["data"]

sim_data.plot_sim_data(data["data"], dist="normal")

# Define Groups for the smooth parameter priors
class VarianceIG(lsl.Group):
    def __init__(
        self, name: str, a: float, b: float, start_value: float = 1.0
    ) -> None:
        a_var = lsl.Var(a, name=f"{name}_a")
        b_var = lsl.Var(b, name=f"{name}_b")

        prior = lsl.Dist(tfjd.InverseGamma, concentration=a_var, scale=b_var)
        tau2 = lsl.Param(start_value, distribution=prior, name=name)
        super().__init__(name=name, a=a_var, b=b_var, tau2=tau2)

class SplineCoef(lsl.Group):
    def __init__(self, name: str, penalty: Array, tau2: lsl.Param) -> None:
        penalty_var = lsl.Var(penalty, name=f"{name}_penalty")

        # we save the rank of the penalty matrix here to use it later in our
        # gibbs kernel
        rank = lsl.Data(np.linalg.matrix_rank(penalty), _name=f"{name}_rank")

        prior = lsl.Dist(
            LieselMultivariateNormalDegenerate.from_penalty,
            loc=0.0,
            var=tau2,
            pen=penalty_var,
        )
        start_value = np.zeros(np.shape(penalty)[-1], np.float32)

        coef = lsl.Param(start_value, distribution=prior, name=name)
        
        super().__init__(name, coef=coef, penalty=penalty_var, tau2=tau2, rank=rank)

class PSpline(lsl.Group):
    def __init__(
        self, name, basis_matrix: Array, penalty: Array, tau2_group: lsl.Group
    ) -> None:
        coef_group = SplineCoef(
            name=f"{name}_coef", penalty=penalty, tau2=tau2_group["tau2"]
        )

        basis_matrix = lsl.Obs(basis_matrix, name="basis_matrix")
        smooth = lsl.Var(
            lsl.Calc(jnp.dot, basis_matrix, coef_group["coef"]), name=name
        )

        group_vars = coef_group.nodes_and_vars | tau2_group.nodes_and_vars

        super().__init__(
            name=name,
            basis_matrix=basis_matrix,
            smooth=smooth,
            **group_vars
        )

# Set up model in liesel
# Fixed parameter prior
beta_loc = lsl.Var(0.0, name="beta_loc")
beta_scale = lsl.Var(100.0, name="beta_scale")

# Set up the fixed parameters
beta_dist = lsl.Dist(tfjd.Normal, loc=beta_loc, scale=beta_scale)
beta = lsl.Param(value=np.array([0.0]), distribution=beta_dist, name="beta")

# Set up the smooth parameters
tau2_group = VarianceIG(name="tau2", a=1.0, b=0.0005)

penalty = X.smooth_pen_mat_cent[0]
smooth_x = PSpline(name="smooth_x", basis_matrix=X.design_mat_cent[1], penalty=penalty, tau2_group=tau2_group)

# Set up the scale 
sigma_a = lsl.Var(0.01, name="sigma_a")
sigma_b = lsl.Var(0.01, name="sigma_b")

sigma_dist = lsl.Dist(tfjd.InverseGamma, concentration=sigma_a, scale=sigma_b)
sigma = lsl.Param(value=10.0, distribution=sigma_dist, name="sigma")

Z = lsl.Obs(X.fixed_data, name="Z")

lpred_loc_fn = lambda z, beta, smooth_1: jnp.dot(z, beta) + smooth_1
lpred_loc_calc = lsl.Calc(lpred_loc_fn, z=Z, beta=beta, smooth_1=smooth_x["smooth"])

lpred_loc = lsl.Var(lpred_loc_calc, name="lpred_loc")

response_dist = lsl.Dist(tfjd.Normal, loc=lpred_loc, scale=sigma)
response = lsl.Var(y.to_numpy(), distribution=response_dist, name="response")