"""
Black-Box-Variational-Inference.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.tree_util import tree_flatten, tree_unflatten
from jax.example_libraries import optimizers

import numpy as np

import tensorflow as tf
import tensorflow_probability.substrates.jax.distributions as tfjd
import tensorflow_probability.substrates.numpy.distributions as tfnd

import matplotlib.pyplot as plt

from typing import List, Dict, Tuple

from .variational import Variational

from ..model.model import (
    Model,
    Lpred,
    Param,
    calc,
    Array,
    Any,
    Distribution)

class Bbvi:
    """
    Estimation algorithm.
    """

    def __init__(self, Model: Model, num_samples: int, num_iterations: int, seed: Any) -> None:
        self.Model = Model
        self.tree = Model.tree
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.seed = seed
        self.variational_params = {}
        self.opt_variational_params = {}
        self.samples = {}
        self.ELBO = []

        self.set_variational_params()

    # initialize the variational parameters of the model
    def set_variational_params(self):
        for kw1, input1 in self.Model.y_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                for kw2, input2 in input1.kwinputs.items():
                    self.variational_params[kw2] = self.init_varparam(input2.dim)
            elif isinstance(input1, Param):
                self.variational_params[kw1] = self.init_varparam(input1.dim)

    # initialize the variational parameters
    def init_varparam(self, dim: Any) -> Dict:
        mu = jnp.zeros(dim)
        lower_tri = jnp.tri(dim[0])
        return {"mu": mu, "lower_tri": lower_tri}

    # calculate the logprob of the model
    def logprob(self, sample: Dict) -> Array:
        logprior = []
        params = {}
        for kw1, input1 in self.Model.y_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                arrays = []
                for kw2, input2 in input1.kwinputs.items():
                    if input2.function is not None:
                        sample[kw2] = self.tree["y_dist"][kw1][kw2]["bijector"](sample[kw2])
                    logprior.append(jnp.sum(self.tree["y_dist"][kw1][kw2]["dist"].log_prob(sample[kw2])))
                    arrays.append(sample[kw2])

                flat_arrays, _ = tree_flatten(arrays)

                beta = jnp.concatenate(flat_arrays, dtype=jnp.float32)

                nu = calc(self.tree["y_dist"][kw1]["fixed"], beta)
                if input1.function is not None:
                    params[kw1] = self.tree["y_dist"][kw1]["bijector"](nu)
                else:
                    params[kw1] = nu

            elif isinstance(input1, Param):
                if input1.function is not None:
                    sample[kw1] = self.tree["y_dist"][kw1]["bijector"](sample[kw1])
                logprior.append(jnp.sum(self.tree["y_dist"][kw1]["dist"].log_prob(sample[kw1])))
                params[kw1] = sample[kw1]

        logprior = jnp.asarray(logprior)

        loglik = self.tree["y_dist"]["dist"](**params).log_prob(self.tree["y"])

        return jnp.sum(loglik) + jnp.sum(logprior)

    # function to calculate the logprob for all samples and update the graph
    def pass_samples(self, samples: Dict) -> Array:

        def compute_logprob(i):
            sample = {key: value[i] for key, value in samples.items()}
            return self.logprob(sample=sample)

        logprobs = jax.vmap(compute_logprob)(jnp.arange(self.num_samples))

        return logprobs

    # define function that estimates the evidence lower bound
    def lower_bound(self, variational_params: Dict) -> Array:
        arrays = []
        for kw in variational_params.keys():
            mu, lower_tri = variational_params[kw]["mu"], variational_params[kw]["lower_tri"]

            self.samples[kw] = tfjd.MultivariateNormalTriL(loc=mu, scale_tril=lower_tri).sample(sample_shape=(self.num_samples,), seed=self.seed)

            e = jnp.mean(tfjd.MultivariateNormalTriL(loc=mu, scale_tril=lower_tri).log_prob(self.samples[kw]))
            arrays.append(e)

        x = jnp.array(arrays, dtype=jnp.float32)

        elbo = jnp.mean(self.pass_samples(self.samples)) - jnp.sum(x)
        return -elbo

    def run_bbvi(self, step_size: float = 0.01) -> Tuple:
        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
        opt_state = opt_init(self.variational_params)

        @jit
        def step(step, opt_state):
            value, grads = jax.value_and_grad(self.lower_bound)(get_params(opt_state))
            opt_state = opt_update(step, grads, opt_state)
            return value, opt_state

        for i in range(self.num_iterations):
            value, opt_state = step(i, opt_state)
            self.ELBO.append(-value)

        self.ELBO = jnp.asarray(self.ELBO, dtype=jnp.float32).ravel()
        self.variational_params = get_params(opt_state)
        self.set_opt_variational_params()
        return value, self.opt_variational_params

    # initialize the variational parameters of the model
    def set_opt_variational_params(self):
        for kw1, input1 in self.Model.y_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                for kw2, input2 in input1.kwinputs.items():
                    self.opt_variational_params[kw2] = {}
                    self.opt_variational_params[kw2]["mu"] = self.variational_params[kw2]["mu"]
                    self.opt_variational_params[kw2]["cov"] = jnp.dot(self.variational_params[kw2]["lower_tri"], self.variational_params[kw2]["lower_tri"].T)
            elif isinstance(input1, Param):
                self.opt_variational_params[kw1] = {}
                self.opt_variational_params[kw1]["mu"] = self.variational_params[kw1]["mu"]
                self.opt_variational_params[kw1]["cov"] = jnp.dot(self.variational_params[kw1]["lower_tri"], self.variational_params[kw1]["lower_tri"].T)

    # plot the progression of the ELBO
    def plot_elbo(self):
        # visulaize the ELBO
        plt.plot(jnp.arange(self.num_iterations), self.ELBO)
        plt.title("Progression of the ELBO")
        plt.xlabel("Iteration")
        plt.ylabel("ELBO")
        plt.show()
