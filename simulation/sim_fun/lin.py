"""
Simulate linear function.
"""

# Dependencies 
import numpy as np
import jax.numpy as jnp 
import jax
import tensorflow_probability.substrates.jax.distributions as tfjd

def lin_fun(x, beta):
    y = beta[0] + beta[1]*x
    return y

def add_noise(f, sigma, key):
    e = tfjd.Normal(loc=jnp.array([0.0]), scale = sigma).sample(sample_shape=f.shape, seed=key)
    y = f + e
    return y

key = jax.random.PRNGKey(27)
key, *subkeys = jax.random.split(key, 3)
x = jax.random.uniform(subkeys[0], (100,), minval=-3, maxval=3)
beta = jnp.array([0.0, 1.0/1.758])

test = lin_fun(x, beta)
y_test = add_noise(test, jnp.array([1.0]), subkeys[1])
print("Function:")
print(jnp.mean(test), jnp.std(test))
print(jnp.quantile(test, q=jnp.array([0.025, 0.975])))
print("Function+Noise:")
print(jnp.mean(y_test), jnp.std(y_test))
print(jnp.quantile(y_test, q=jnp.array([0.025, 0.975])))