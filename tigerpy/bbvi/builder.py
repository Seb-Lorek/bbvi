"""
Black-Box-Variational-Inference.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.example_libraries import optimizers

import numpy as np

import tensorflow_probability.substrates.jax.distributions as tfjd
import tensorflow_probability.substrates.numpy.distributions as tfnd

from typing import List, Dict

from .variational import Variational

from ..model.model import (
    Model,
    Lpred,
    Param,
    Array,
    Any,
    Distribution)

class Bbvi:
    """
    Estimation algorithm.
    """

    def __init__(self, Model: Model, num_samples: int, num_iterations: int, seed: Any) -> None:
        self.Model = Model
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.seed = seed
        self.variational_params = {}
        self.bijectors = {}
        self.samples = {}
        self.ELBO = []

        self.get_params()
        self.get_bijectors()

    # initialize the variational parameters of the model
    def get_params(self):
        for kw1, input1 in self.Model.y_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                for kw2, input2 in input1.kwinputs.items():
                    self.variational_params[kw2] = self.init_varparam(input2.dim)
            elif isinstance(input1, Param):
                self.variational_params[kw1] = self.init_varparam(input1.dim)

    # get the bijectors of the model
    def get_bijectors(self):
        for kw1, input1 in self.Model.y_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                for kw2, input2 in input1.kwinputs.items():
                    self.bijectors[kw2] = input2.function
            elif isinstance(input1, Param):
                self.bijectors[kw1] = input1.function

    # initialize the variational parameters
    def init_varparam(self, dim: Any) -> Dict:
        mu = jnp.zeros(dim)
        cov = jnp.zeros(dim)
        return {"mu": mu, "cov": cov}


    # function to calculate the logprob for all samples and update the graph
    def logprob(self, samples: Dict) -> Array:
        for kw in samples.keys():
            if self.bijectors[kw] is not None:
                samples[kw] = self.bijectors[kw](samples[kw])

        def compute_logprob(i):
            sample = {key: value[i] for key, value in samples.items()}
            return self.Model.update_graph(sample=sample)

        logprobs = jax.vmap(compute_logprob)(jnp.arange(self.num_samples))

        return logprobs

    # define function that estimates the evidence lower bound
    def lower_bound(self, variational_params: Dict) -> Array:
        arrays = []
        for kw in variational_params.keys():
            mu, cov = variational_params[kw]["mu"], jnp.exp(variational_params[kw]["cov"])
            self.samples[kw] = tfjd.MultivariateNormalDiag(loc=jnp.zeros(mu.shape), scale_diag=jnp.ones(cov.shape)).sample(sample_shape=(self.num_samples,), seed=self.seed) * jnp.sqrt(cov) + mu
            e = tfjd.MultivariateNormalDiag(loc=mu, scale_diag=cov).entropy()
            arrays.append(e)

        x = jnp.array(arrays, dtype=jnp.float32)

        return - jnp.sum(x) - jnp.mean(self.logprob(self.samples))

    def run_bbvi(self):
        opt_init, opt_update, get_params = optimizers.adam(step_size=0.05)
        opt_state = opt_init(self.variational_params)

        @jit
        def step(step, opt_state):
            value, grads = jax.value_and_grad(self.lower_bound)(get_params(opt_state))
            opt_state = opt_update(step, grads, opt_state)
            return value, opt_state

        for i in range(self.num_iterations):
            value, opt_state = step(i, opt_state)
            self.ELBO.append(value * (-1))

        self.ELBO = jnp.asarray(self.ELBO, dtype=jnp.float32).ravel()
        return value, get_params(opt_state)
