"""
The degenerate, i.e. rank-deficient, multivariate normal distribution.
"""

from ..utils import (
    dot,
    quad_prod
)

from ..model.observation import Obs

from functools import cached_property
from typing import Any
Array = Any

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfjd
from tensorflow_probability.python.internal import reparameterization

def _rank(eigenvalues: Array, tol: float = 1e-6) -> Array | float:
    """
    Computes the rank of a matrix based on the provided eigenvalues.
    The rank is taken to be the number of non-zero eigenvalues.

    Can handle batches.
    """
    mask = eigenvalues > tol
    rank = jnp.sum(mask, axis=-1)
    return rank

def _log_pdet(eigenvalues: Array, rank: Array | float | None = None,
             tol: float = 1e-6) -> Array | float:
    """
    Computes the log of the pseudo-determinant of a matrix based on the provided
    eigenvalues. If the rank is provided, it is used to select the non-zero eigenvalues.
    If the rank is not provided, it is computed by counting the non-zero eigenvalues. An
    eigenvalue is deemed to be non-zero if it is greater than the numerical tolerance
    ``tol``.

    Can handle batches.
    """

    if rank is None:
        mask = eigenvalues > tol
    else:
        max_index = eigenvalues.shape[-1] - rank

        def fn(i, x):
            return x.at[..., i].set(i >= max_index)

        mask = jax.lax.fori_loop(0, eigenvalues.shape[-1], fn, eigenvalues)

    selected = jnp.where(mask, eigenvalues, 1.0)
    log_pdet = jnp.sum(jnp.log(selected), axis=-1)
    return log_pdet

class MulitvariateNormalDegenerate(tfjd.Distribution):

    def __init__(self,
                 loc: Array,
                 scale: Array,
                 pen: Array,
                 validate_args: bool = False,
                 allow_nan_stats: bool = True,
                 name: str = "MultivariateNormalDegenerate",
                 tol: float = 1e-6):

        parameters = dict(locals())

        self._loc = jnp.atleast_1d(loc)
        self._scale = jnp.atleast_1d(scale)
        self._tol = tol
        self._pen = pen
        self._prec = pen / (self._scale ** 2)

        eigenvals = jnp.linalg.eigvalsh(pen)
        self._rank = _rank(eigenvalues=eigenvals, tol=tol)
        log_pdet_pen = _log_pdet(eigenvalues=eigenvals, rank=self._rank)
        self._log_pdet = log_pdet_pen - self._rank * jnp.log(self._scale ** 2)

        super().__init__(dtype=pen.dtype,
                         reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
                         validate_args=validate_args,
                         allow_nan_stats=allow_nan_stats,
                         parameters=parameters,
                         name=name)

    @cached_property
    def eigenvals(self) -> Array:
        """Eigenvalues of the distribution's precision matrices."""
        return jnp.linalg.eigvalsh(self._prec)

    @cached_property
    def rank(self) -> Array | float:
        """Ranks of the distribution's precision matrices."""
        eigenvals = self.eigenvals
        return _rank(eigenvals, tol=self._tol)

    @cached_property
    def log_pdet(self) -> Array | float:
        """Log-pseudo-determinants of the distribution's precision matrices."""
        eigenvals = self.eigenvals
        return _log_pdet(eigenvals, self._rank, tol=self._tol)

    @property
    def prec(self) -> Array:
        """Precision matrices."""
        return self._prec

    @property
    def loc(self) -> Array:
        """Locations."""
        return self._loc

    def log_prob(self, x: Array) -> Array | float:
        x_centerd = x - self._loc

        prob1 = - quad_prod(self._prec, x_centerd)
        prob2 = self._rank * jnp.log(2 * jnp.pi) - self._log_pdet
        return 0.5 * (prob1 - prob2)
