"""
Test functions.
"""



# import the lower bound
def lower_bound(self, variational_params: Dict) -> Array:
    mus = jnp.stack([variational_params[kw]["mu"] for kw in variational_params.keys()])
    covs = jnp.stack([jnp.exp(variational_params[kw]["cov"]) for kw in variational_params.keys()])

    # Calculate samples
    self.samples = jax.random.multivariate_normal(self.seed, jnp.zeros_like(mus), jnp.diag(jnp.ones_like(mus)))

    # Calculate entropies
    entropies = 0.5 * jnp.sum(jnp.log(2 * jnp.pi * jnp.exp(1) * covs), axis=1)

    return -jnp.sum(entropies) - jnp.mean(self.logprob(self.samples))


# list comprehension
l = [self.Model.update_graph(sample={key: value[i] for key, value in samples.items()}) for i in range(self.num_samples)]

# improve the logprob to allow for vmap
def logprob(self, samples: Dict) -> Array:
    sample_fn = lambda i: {key: (value[i] if self.bijectors[key] is None else self.bijectors[key](value[i])) for key, value in samples.items()}
    sample_indices = jnp.arange(self.num_samples)
    samples_batched = vmap(sample_fn)(sample_indices)

    update_graph_batched = jnp.vectorize(self.Model.update_graph, signature="(n)->()")
     values_batched = update_graph_batched(samples_batched)

    return jnp.array(values_batched, dtype=jnp.float32)

# calculate the gradient
# pass this gradient to ADAM
def gradient(self, variational_params: Dict) -> Dict:
    gradient_func = grad(self.lower_bound)
    return gradient_func(variational_params)
