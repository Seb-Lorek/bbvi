"""
Variational distribution.
"""

import jax.numpy as jnp

from ..model.model import (
    Dist,
    Array,
    Any,
    Distribution)

# pass all variational distributions.
class Variational:
    """
    Variational distribution class.
    """

    def __init__(self, **kwinputs: Distribution) -> None:
        self.kwinputs = kwinputs
        self.log_prob = self.logprob()

    def logprob(self):
        x = jnp.array([], dtype=jnp.float32)
        for kw, input in self.kwinputs.items():
             x = jnp.append(x, input.log_prob)
        return jnp.sum(x)

# define the variational paramters.
class VarParam:
    """
    Variational parameters.
    """

    def __init__(self, value: Array, name: str = "") -> None:
        self.value = value
        self.name = name

    def __repr__(self) -> str:
        return f'{type(self).__name__}(name="{self.name}")'

# class to set the variational distribution.
class Dist:
    """
    Variational distribution.
    """

    def __init__(self, distribution: Distribution, name: str = "", **kwinputs: Any):
        self.distribution = distribution
        self.name = name
        self.kwinputs = kwinputs

    def init_dist(self) -> Distribution:
        """
        Initialize variational distribution of parameter.

        Returns:
            Dist: A tensorflow probability.
        """

        kwargs = {kw: input.value for kw, input in self.kwinputs.items()}
        dist = self.distribution(**kwargs)
        return dist

# class to define a full variational factor
class Var:
    """
    Variational factor.
    """

    def __init__(self, value: Array, distribution: Distribution, name: str = "") -> None:
        self.value = jnp.atleast_1d(value)
        self.dim = self.value.shape
        self.distribution = distribution
        self.name = name
        self.log_prob = self.logprob(value=self.value)

    # update the logprob if values have changed
    def logprob(self, value: Array) -> Array:
        return self.distribution.init_dist().log_prob(value)
